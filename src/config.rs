use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Deserialize)]
pub struct ProcessingConfig {
    pub batch_hours: usize,
    pub compression: usize,
    pub chunk_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct PathsConfig {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub temp_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct PercentilesConfig {
    pub values: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct GdalConfig {
    pub predictor: String,
    pub compress: String,
    pub tiled: String,
    pub bigtiff: String,
}

#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
}

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub app: AppConfig,
    pub processing: ProcessingConfig,
    pub paths: PathsConfig,
    pub percentiles: PercentilesConfig,
    pub gdal: GdalConfig,
    pub logging: LoggingConfig,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let run_mode = std::env::var("RUN_MODE").unwrap_or_else(|_| "development".into());

        let s = Config::builder()
            // 首先加载默认配置
            .add_source(File::with_name("settings/default"))
            // 然后加载环境特定的配置
            .add_source(File::with_name(&format!("settings/{}", run_mode)).required(false))
            // 最后加载环境变量（可选）
            .add_source(Environment::with_prefix("APP"))
            .build()?;

        // 反序列化配置
        s.try_deserialize()
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        // 验证路径
        if !self.paths.input_dir.exists() {
            return Err(ConfigError::NotFound("输入目录不存在".into()));
        }
        
        // 验证处理参数
        if self.processing.batch_hours == 0 {
            return Err(ConfigError::Invalid("batch_hours 必须大于 0".into()));
        }
        
        // 验证百分位数
        for &p in &self.percentiles.values {
            if !(0.0..=100.0).contains(&p) {
                return Err(ConfigError::Invalid("百分位数必须在 0-100 之间".into()));
            }
        }
        
        Ok(())
    }
} 