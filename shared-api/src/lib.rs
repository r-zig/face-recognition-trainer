use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use compreface_contracts::CompreFaceConfig;
use double_take_contracts::DoubleTakeConfig;
use futures::StreamExt;
use std::{
    fmt::{Display, Formatter},
    future::Future,
    path::PathBuf,
};
use stream_utils::{BufferUntilCondition, RecursiveFileStream};
use tokio::{fs, sync::mpsc::Sender};

pub mod utils;
/// Trainer trait
/// This trait is used to train a model with a set of images and a name
/// The function send instructions to the destination to train the model, but the train itself is async
#[async_trait]
pub trait Trainer {
    async fn send_to_train(&self, name: &str, files: Vec<PathBuf>) -> anyhow::Result<TrainResult>;
}

#[async_trait]
pub trait TrainLogic {
    async fn train(
        &self,
        config: &Configuration,
        tx: Sender<ProgressReporter<TrainResult>>,
    ) -> anyhow::Result<TrainResult>;
}

#[async_trait]
impl<F, Fut> TrainLogic for F
where
    F: Fn(&Configuration, Sender<ProgressReporter<TrainResult>>) -> Fut + Send + Sync,
    Fut: Future<Output = anyhow::Result<TrainResult>> + Send,
{
    async fn train(
        &self,
        config: &Configuration,
        tx: Sender<ProgressReporter<TrainResult>>,
    ) -> anyhow::Result<TrainResult> {
        (self)(config, tx).await
    }
}

#[derive(Clone, Debug)]
pub struct TrainResult {}

impl Display for TrainResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Training completed")
    }
}
/// Recognize trait
/// This trait is used to recognize faces in a set of images and a given name
#[async_trait]
pub trait Recognizer {
    async fn recognize(&self, name: &str, files: Vec<PathBuf>) -> anyhow::Result<RecognizeResult>;
}

/// RecognizeResult struct to hold the result of the recognition
#[derive(Debug)]
pub struct RecognizeResult {
    /// The total number of faces that were queried
    pub total_count: usize,

    /// The number of faces that were recognized successfully
    pub success_count: usize,

    /// The number of faces that did not have a match
    pub failure_count: usize,

    /// The list of faces that were not recognized
    pub unrecognized: Vec<PathBuf>,
}

impl Display for RecognizeResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "%success: {}, Total: {}, Success: {}, Failure: {}, Unrecognized: {}",
            self.get_success_percentage(),
            self.total_count,
            self.success_count,
            self.failure_count,
            self.unrecognized.len()
        )
    }
}
impl RecognizeResult {
    pub fn with_capacity(total_count: usize) -> Self {
        RecognizeResult {
            total_count,
            success_count: 0,
            failure_count: 0,
            unrecognized: Vec::with_capacity(total_count),
        }
    }

    /// add other RecognizeResult to this one
    /// This is used to merge the results of multiple RecognizeResult
    pub fn add(&mut self, other: RecognizeResult) {
        self.total_count += other.total_count;
        self.success_count += other.success_count;
        self.failure_count += other.failure_count;
        self.unrecognized.extend(other.unrecognized);
    }

    /// get percentage of the success recognition
    pub fn get_success_percentage(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        (self.success_count as f64 / self.total_count as f64) * 100.0
    }
}

impl Clone for RecognizeResult {
    fn clone(&self) -> Self {
        RecognizeResult {
            total_count: self.total_count,
            success_count: self.success_count,
            failure_count: self.failure_count,
            unrecognized: self.unrecognized.clone(),
        }
    }
}
/// ProgressReporter enum to report the progress of the training or recognition operation
pub enum ProgressReporter<T>
where
    T: Clone + std::marker::Sync + std::marker::Send + 'static,
{
    /// Increase the progress fill by the given value
    Increase(u64),
    /// Increase the progress length by the given value
    IncreaseLength(u64),
    /// Set the progress message
    Message(String),

    /// Current custom value
    StructedMessage(T),
    /// Finish the progress with the given message
    FinishWithMessage(String),
}

#[derive(Debug, clap::Parser, Clone)]
#[clap(name = "face-recognition-trainer")]
pub struct Configuration {
    /// The client type to use, Compreface or DoubleTake
    #[arg(long, value_enum)]
    pub client_type: ClientType,

    /// The client mode to run, train or recognize
    /// The default value is train
    #[clap(long, short, default_value = "train")]
    pub client_mode: ClientMode,

    #[clap(flatten)]
    pub double_take: Option<DoubleTakeConfig>,

    #[clap(flatten)]
    pub compreface: Option<CompreFaceConfig>,

    /// The data set root folder with the face images
    #[clap(long, env = "DATASET_PATH", default_value = ".")]
    pub dataset_path: String,

    /// The maximum size of the request to send to the double-take service
    /// The service will be called when the total size of the files content reaches this size
    /// The default value is 10MB
    #[clap(long, env = "MAX_REQUEST_SIZE", default_value = "10485760")]
    pub max_request_size: u64,

    /// Optional name for all scanned faces
    /// When set, it will ignore the folder name per image and use this name for all faces
    #[clap(long, env = "OVERRIDE_TRAINED_NAME")]
    pub override_trained_name: Option<String>,
}

impl Configuration {
    pub fn get() -> Result<Self, String> {
        let config = Configuration::parse();
        match config.client_type {
            ClientType::Compreface => {
                if config.compreface.is_none() {
                    return Err(
                        "--compreface-url & compreface-api-key are required when client_mode is CompreFace".into(),
                    );
                }
            }
            ClientType::DoubleTake => {
                if config.double_take.is_none() {
                    return Err(
                        "--doubletake-url is required when client_mode is DoubleTake".into(),
                    );
                }
            }
        }
        Ok(config)
    }
}
#[derive(ValueEnum, Clone, Debug)]
pub enum ClientType {
    Compreface,
    DoubleTake,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum ClientMode {
    Train,
    Recognize,
}

pub async fn process_files<T, F, Fut>(
    config: &Configuration,
    tx: Sender<ProgressReporter<T>>,
    api_action: F,
) -> anyhow::Result<()>
where
    F: Fn(String, Vec<PathBuf>) -> Fut + Send + Sync,
    Fut: Future<Output = anyhow::Result<()>> + Send,
    T: Clone + std::marker::Sync + std::marker::Send + 'static,
{
    tx.send(ProgressReporter::Message(format!(
        "Start processing directory: {}",
        &config.dataset_path
    )))
    .await?;

    let files = RecursiveFileStream::new(&config.dataset_path);
    let mut files_groups = BufferUntilCondition::new(files, |path| path.as_ref().unwrap().is_dir());

    while let Some(group) = files_groups.next().await {
        let name = match config.override_trained_name {
            Some(ref name) => name.to_string(),
            None => utils::get_directory_name(&group)?,
        };

        tx.send(ProgressReporter::IncreaseLength(group.len() as u64))
            .await?;
        tx.send(ProgressReporter::Message(format!(
            "processing directory: {}",
            &name
        )))
        .await?;

        let mut files_content: Vec<PathBuf> = Vec::new();
        let mut total_size = 0;

        for path in group.into_iter() {
            let path_buf = path?;
            if path_buf.is_dir() {
                tx.send(ProgressReporter::Message(format!(
                    "{}",
                    path_buf.file_stem().unwrap().to_string_lossy()
                )))
                .await?;
                continue;
            }

            if !utils::is_image(&path_buf) {
                continue;
            }

            let file_len = fs::metadata(path_buf.clone()).await?.len();
            if total_size + file_len > config.max_request_size {
                let current_items = files_content.len() as u64;
                api_action(name.clone(), files_content.clone()).await?;
                tx.send(ProgressReporter::Increase(current_items)).await?;
                files_content.clear();
                total_size = 0;
            }

            total_size += file_len;
            files_content.push(path_buf);
        }

        if !files_content.is_empty() {
            let current_items = files_content.len() as u64;
            api_action(name, files_content).await?;
            tx.send(ProgressReporter::Increase(current_items)).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {}
