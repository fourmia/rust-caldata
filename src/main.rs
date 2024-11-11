use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tdigest::TDigest;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use gdal;
use log::{info, debug};
use env_logger::Env;
use serde::{Serialize, Deserialize};
use gdal::raster::RasterCreationOption;
use std::fs;
use bincode::{serialize, deserialize};
mod config;
use config::Settings;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
struct GridCoord {
    x: usize,
    y: usize,
}

// 为 TDigest 实现自定义序列化
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializableTDigest {
    centroids: Vec<(f64, f64)>,  // (mean, weight)
    count: usize,
    max_size: usize,
}

impl From<&TDigest> for SerializableTDigest {
    fn from(digest: &TDigest) -> Self {
        let centroids: Vec<(f64, f64)> = digest.into_centroids()
            .iter()
            .map(|c| (c.mean(), c.weight()))
            .collect();
        
        SerializableTDigest {
            centroids,
            count: digest.count(),
            max_size: digest.size(),
        }
    }
}

impl From<SerializableTDigest> for TDigest {
    fn from(serializable: SerializableTDigest) -> Self {
        let mut digest = TDigest::new_with_size(serializable.max_size);
        for (mean, weight) in serializable.centroids {
            digest = digest.merge_unsorted(vec![mean; weight as usize]);
        }
        digest
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DataBlock {
    digests: HashMap<GridCoord, SerializableTDigest>,
    modified: bool,
}

impl DataBlock {
    fn new(settings: &Settings) -> Self {
        Self {
            digests: HashMap::with_capacity(settings.processing.chunk_size),
            modified: false,
        }
    }

    fn update_digest(&mut self, coord: GridCoord, values: &[f64], compression: usize) {
        if !values.is_empty() {
            let mut digest = self.digests.get(&coord)
                .map(|d| TDigest::from(d.clone()))
                .unwrap_or_else(|| TDigest::new_with_size(compression));
            
            let old_count = digest.count() as f64;
            digest = digest.merge_sorted(values.to_vec());
            
            self.digests.insert(coord, SerializableTDigest::from(&digest));
            self.modified = true;
        }
    }

    fn merge_digest(&mut self, coord: GridCoord, other_digest: TDigest) {
        let mut digest = self.digests.get(&coord)
            .map(|d| TDigest::from(d.clone()))
            .unwrap_or_else(|| TDigest::new_with_size(digest.size()));
        
        digest = TDigest::merge_digests(vec![digest, other_digest]);
        self.digests.insert(coord, SerializableTDigest::from(&digest));
        self.modified = true;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProcessingState {
    processed_files: Vec<PathBuf>,
    last_block: Option<DataBlock>,
    dimensions: (usize, usize),
}

struct GridProcessor {
    blocks: HashMap<GridCoord, DataBlock>,
    temp_dir: PathBuf,
    progress: MultiProgress,
    width: usize,
    height: usize,
}

impl GridProcessor {
    fn new(temp_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&temp_dir)?;
        Ok(Self {
            blocks: HashMap::new(),
            temp_dir,
            progress: MultiProgress::new(),
            width: 0,
            height: 0,
        })
    }

    fn set_dimensions(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
    }

    fn process_single_file(&mut self, file: &Path) -> Result<()> {
        debug!("开始处理文件: {:?}", file);
        let dataset = gdal::Dataset::open(file)?;
        let band = dataset.rasterband(1)?;
        
        let no_data_value = band.no_data_value().unwrap_or(f64::NAN);
        let (width, height) = dataset.raster_size();
        self.width = width as usize;
        self.height = height as usize;

        let data = band.read_as::<f64>(
            (0, 0),
            (width, height),
            (self.width, self.height),
            None,
        )?;

        // 创建或获取现有的数据块，使用特定的 GridCoord 作为键
        let block_key = GridCoord { x: 0, y: 0 };  // 使用固定的键
        let block = self.blocks.entry(block_key).or_insert_with(DataBlock::new);

        // 按位置收集数据
        for y in 0..height as usize {
            for x in 0..width as usize {
                let value = data.data[y * width as usize + x];
                if value != no_data_value && value.is_finite() && value > -10000.0 && value < 100000000.0 {
                    let coord = GridCoord { x, y };
                    block.update_digest(coord, &[value], 100);
                }
            }
        }

        // 打印一些统计信息
        for (coord, digest) in &block.digests {
            let count = digest.count() as f64;  // 转换为 f64
            if count > 0.0 {
                let min = digest.estimate_quantile(0.0);
                let max = digest.estimate_quantile(1.0);
                debug!("位置 ({}, {}): 时间序列长度 = {:.0}, 范围 = [{:.2}, {:.2}]", 
                    coord.x, coord.y, count, min, max);
            }
        }

        Ok(())
    }

    fn calculate_percentiles(&self, percentiles: &[f64]) -> Result<Vec<Vec<Vec<f64>>>> {
        let mut results = vec![vec![vec![0.0; self.width]; self.height]; percentiles.len()];
        
        let pb = self.progress.add(ProgressBar::new((self.height * self.width) as u64));
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:50.cyan/blue}] {pos}/{len} 计算百分位数")
            .unwrap());

        // 使用相同的 block_key
        let block_key = GridCoord { x: 0, y: 0 };

        // 对每个网格点计算百分位数
        for y in 0..self.height {
            for x in 0..self.width {
                let coord = GridCoord { x, y };
                
                if let Some(block) = self.blocks.get(&block_key) {
                    if let Some(digest) = block.digests.get(&coord) {
                        let count = digest.count() as f64;  // 转换为 f64
                        if count > 0.0 {
                            debug!("位置 ({}, {}): 时间序列长度 = {:.0}", x, y, count);
                            
                            for (i, &p) in percentiles.iter().enumerate() {
                                let percentile = p / 100.0;
                                let value = digest.estimate_quantile(percentile);
                                results[i][y][x] = value;
                                
                                debug!("位置 ({}, {}), 百分位数 {:.1}% = {}", x, y, p, value);
                            }
                        } else {
                            for i in 0..percentiles.len() {
                                results[i][y][x] = f64::NAN;
                            }
                        }
                    } else {
                        for i in 0..percentiles.len() {
                            results[i][y][x] = f64::NAN;
                        }
                    }
                } else {
                    for i in 0..percentiles.len() {
                        results[i][y][x] = f64::NAN;
                    }
                }
                pb.inc(1);
            }
        }

        pb.finish_with_message("百分位数计算完成");
        
        // 打印结果统计信息
        for (i, &p) in percentiles.iter().enumerate() {
            let values: Vec<f64> = results[i].iter()
                .flatten()
                .filter(|&&x| x.is_finite())
                .copied()
                .collect();
            
            if !values.is_empty() {
                let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                info!("百分位数 {:.1}%: 有效值数量 = {}, 范围 = [{:.2}, {:.2}], 平均值 = {:.2}", 
                    p, values.len(), min, max, mean);
            } else {
                info!("百分位数 {:.1}%: 没有有效值", p);
            }
        }

        Ok(results)
    }

    // 添加保存状态的方法
    fn save_state(&self, processed_files: &[PathBuf]) -> Result<()> {
        let state = ProcessingState {
            processed_files: processed_files.to_vec(),
            last_block: self.blocks.get(&GridCoord { x: 0, y: 0 }).cloned(),
            dimensions: (self.width, self.height),
        };

        let state_path = self.temp_dir.join("processing_state.bin");
        let serialized = serialize(&state)?;
        fs::write(&state_path, serialized)?;
        
        info!("保存处理状态到: {:?}, 已处理 {} 个文件", state_path, processed_files.len());
        Ok(())
    }

    // 添加加载状态的方法
    fn load_state(&mut self) -> Result<Option<ProcessingState>> {
        let state_path = self.temp_dir.join("processing_state.bin");
        if !state_path.exists() {
            return Ok(None);
        }

        let data = fs::read(&state_path)?;
        let mut state: ProcessingState = deserialize(&data)?;
        
        // 恢复状态
        if let Some(ref block) = state.last_block {  // 使用引用而不是移动
            self.blocks.insert(GridCoord { x: 0, y: 0 }, block.clone());
        }
        self.width = state.dimensions.0;
        self.height = state.dimensions.1;

        info!("加载处理状态: 已处理 {} 个文件", state.processed_files.len());
        Ok(Some(state))
    }

    // 修改批处理方法以支持断点续传
    fn process_file_batch(&mut self, files: &[PathBuf], checkpoint_interval: usize) -> Result<()> {
        let pb = self.progress.add(ProgressBar::new(files.len() as u64));
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:50.cyan/blue}] {pos}/{len} {msg}")
            .unwrap());

        // 设置维度（如果还没设置）
        if self.width == 0 || self.height == 0 {
            if let Some(first_file) = files.first() {
                let dataset = gdal::Dataset::open(first_file)?;
                let (width, height) = dataset.raster_size();
                self.set_dimensions(width as usize, height as usize);
            }
        }

        let mut processed_files = Vec::new();
        for (i, file) in files.iter().enumerate() {
            pb.set_message(format!("处理文件: {}", file.file_name().unwrap().to_string_lossy()));
            self.process_single_file(file)?;
            processed_files.push(file.clone());
            
            // 定期保存状态
            if (i + 1) % checkpoint_interval == 0 {
                self.save_state(&processed_files)?;
            }
            
            pb.inc(1);
        }

        // 最后一次保存状态
        if !processed_files.is_empty() {
            self.save_state(&processed_files)?;
        }

        pb.finish_with_message("批次处理完成");
        Ok(())
    }

    fn save_results(&self, results: Vec<Vec<Vec<f64>>>, percentiles: &[f64], output_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;
        
        let pb = self.progress.add(ProgressBar::new(percentiles.len() as u64));
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:50.cyan/blue}] {pos}/{len} 保存结果")
            .unwrap());

        // 获取第一个输入文件作为参考
        let input_dir = PathBuf::from("input");
        let first_file = std::fs::read_dir(&input_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .find(|path| path.extension().and_then(|s| s.to_str()) == Some("tif"))
            .ok_or("No input files found")?;

        // 读取源文件以获取地理参考信息
        let src_dataset = gdal::Dataset::open(&first_file)?;
        let transform = src_dataset.geo_transform()?;
        let projection = src_dataset.projection();

        for (i, &percentile) in percentiles.iter().enumerate() {
            let filename = format!("percentile_{:.0}.tif", percentile);
            let output_path = output_dir.join(filename);
            
            // 创建新的数据集
            let driver = gdal::Driver::get("GTiff")?;
            
            // 修改 GDAL 选项，移除 PREDICTOR
            let options = [
                RasterCreationOption {
                    key: "COMPRESS",
                    value: "DEFLATE"
                },
                RasterCreationOption {
                    key: "TILED",
                    value: "YES"
                },
                RasterCreationOption {
                    key: "BIGTIFF",
                    value: "YES"
                }
            ];

            let mut dataset = driver.create_with_band_type_with_options::<f64, _>(
                &output_path,
                self.width as isize,
                self.height as isize,
                1,
                &options,
            )?;

            // 设置地理参考信息
            dataset.set_geo_transform(&transform)?;
            dataset.set_projection(&projection)?;

            // 写入数据
            let mut band = dataset.rasterband(1)?;
            let data = results[i].clone().into_iter().flatten()
                .map(|x| if x.is_finite() && x > -100000.0 && x < 100000000.0 { x } else { f64::NAN })
                .collect::<Vec<_>>();

            let buffer = gdal::raster::Buffer::new((self.width, self.height), data);
            band.write((0, 0), (self.width, self.height), &buffer)?;

            // 设置 NoData 值
            band.set_no_data_value(f64::NAN)?;

            pb.inc(1);
            info!("已保存百分位数 {:.0} 的结果到 {:?}", percentile, output_path);
        }

        pb.finish_with_message("结果保存完成");
        Ok(())
    }
}

fn main() -> Result<()> {
    let settings = Settings::new().map_err(|e| {
        eprintln!("配置加载失败: {}", e);
        e
    })?;
    
    // 验证配置值
    if settings.processing.batch_hours == 0 {
        return Err("batch_hours 不能为 0".into());
    }
    
    // 加载配置
    let settings = Settings::new()?;
    
    // 设置日志级别
    env_logger::Builder::from_env(Env::default().default_filter_or(&settings.logging.level))
        .format_timestamp_millis()
        .init();

    info!("程序启动: {}, 版本: {}", settings.app.name, settings.app.version);
    info!("使用 {} 个线程", num_cpus::get());
    
    // 使用配置中的路径
    let processor = GridProcessor::new(settings.paths.temp_dir.clone())?;
    
    // 使用配置中的 GDAL 选项
    let gdal_options = [
        RasterCreationOption {
            key: "COMPRESS",
            value: &settings.gdal.compress
        },
        RasterCreationOption {
            key: "TILED",
            value: &settings.gdal.tiled
        },
        RasterCreationOption {
            key: "BIGTIFF",
            value: &settings.gdal.bigtiff
        }
    ];
    
    // 获取并排序所有输入文件
    let mut files: Vec<PathBuf> = std::fs::read_dir(&settings.paths.input_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("tif"))
        .collect();
    files.sort();

    info!("找到 {} 个输入文件", files.len());

    // 加载之前的处理状态
    let processed_files = if let Some(state) = processor.load_state()? {
        state.processed_files
    } else {
        Vec::new()
    };

    // 找出未处理的文件
    let remaining_files: Vec<PathBuf> = files.into_iter()
        .filter(|f| !processed_files.contains(f))
        .collect();

    info!("已处理 {} 个文件，剩余 {} 个文件待处理", 
        processed_files.len(), remaining_files.len());

    // 按批次处理剩余文件
    const CHECKPOINT_INTERVAL: usize = 100;  // 每处理100个文件保存一次状态
    for chunk in remaining_files.chunks(BATCH_HOURS) {
        processor.process_file_batch(chunk, CHECKPOINT_INTERVAL)?;
    }

    // 计算百分位数
    info!("开始计算百分位数");
    let results = processor.calculate_percentiles(&settings.percentiles.values)?;
    
    info!("始保存结果");
    processor.save_results(results, &settings.percentiles.values, &settings.paths.output_dir)?;

    // 处理完成后删除状态文件
    let state_path = settings.paths.temp_dir.join("processing_state.bin");
    if state_path.exists() {
        fs::remove_file(state_path)?;
        info!("清理临时状态文件");
    }

    info!("程序正常结束");
    Ok(())
}

