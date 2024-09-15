use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use compreface_contracts::CompreFaceConfig;
use double_take_contracts::DoubleTakeConfig;
use futures::StreamExt;
use serde::Deserialize;
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
    async fn send_to_train(
        &self,
        name: &str,
        files: Vec<PathBuf>,
        progress_reporter_tx: Sender<ProgressReporter<FaceProcessingResult>>,
    ) -> anyhow::Result<FaceProcessingResult>;
}

#[async_trait]
pub trait TrainLogic {
    async fn train(
        &self,
        config: &Configuration,
        tx: Sender<ProgressReporter<FaceProcessingResult>>,
    ) -> anyhow::Result<FaceProcessingResult>;
}

#[async_trait]
impl<F, Fut> TrainLogic for F
where
    F: Fn(&Configuration, Sender<ProgressReporter<FaceProcessingResult>>) -> Fut + Send + Sync,
    Fut: Future<Output = anyhow::Result<FaceProcessingResult>> + Send,
{
    async fn train(
        &self,
        config: &Configuration,
        tx: Sender<ProgressReporter<FaceProcessingResult>>,
    ) -> anyhow::Result<FaceProcessingResult> {
        (self)(config, tx).await
    }
}

/// represent the progress of the process
pub trait ProcessProgress {
    fn get_total_count(&self) -> usize;
    fn get_success_count(&self) -> usize;
}

/// Recognize trait
/// This trait is used to recognize faces in a set of images and a given name
#[async_trait]
pub trait Recognizer {
    async fn recognize(
        &self,
        name: &str,
        files: Vec<PathBuf>,
        progress_reporter_tx: Sender<ProgressReporter<FaceProcessingResult>>,
    ) -> anyhow::Result<FaceProcessingResult>;
}

/// RecognizeResult struct to hold the result of the recognition
#[derive(Debug)]
pub struct FaceProcessingResult {
    /// The total number of faces that were queried
    pub total_count: usize,

    /// The number of faces that were recognized successfully
    pub success_count: usize,

    /// The number of faces that did not have a match
    pub failure_count: usize,

    /// The list of faces that were not recognized because of a match
    pub failure_faces: Vec<FailureFace>,

    /// The number of faces that was unable to be processed (for example, when api query fails)
    pub missed_count: usize,

    /// The list of faces that were not recognized because of an error
    pub missed_faces: Vec<PathBuf>,

    pub context: String,
}

#[derive(Debug, Clone)]
pub enum FailureFace {
    Train(PathBuf),              // For training mode, only the path is relevant
    Recognize(FaceWithMetadata), // For recognition mode, include the extra struct
}

#[derive(Debug, Clone)]
pub struct FaceWithMetadata {
    /// The path to the image file
    pub path: PathBuf,
    /// one or more subjects that were recognized for this face
    pub subjects: Vec<Subject>,
}

impl Display for FaceProcessingResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} Total: {}, Success: {}, Failure: {}, missing: {}",
            self.context,
            self.total_count,
            self.success_count,
            self.failure_count,
            self.missed_count
        )
    }
}
impl FaceProcessingResult {
    pub fn with_context(context: String) -> Self {
        FaceProcessingResult {
            total_count: 0,
            success_count: 0,
            failure_count: 0,
            failure_faces: Vec::new(),
            missed_count: 0,
            missed_faces: Vec::new(),
            context,
        }
    }

    /// add other RecognizeResult to this one
    /// This is used to merge the results of multiple RecognizeResult
    pub fn add(&mut self, other: FaceProcessingResult) {
        self.total_count += other.total_count;
        self.success_count += other.success_count;
        self.failure_count += other.failure_count;
        self.failure_faces.extend(other.failure_faces);
        self.missed_count += other.missed_count;
        self.missed_faces.extend(other.missed_faces);
    }
}

impl ProcessProgress for FaceProcessingResult {
    fn get_total_count(&self) -> usize {
        self.total_count
    }

    fn get_success_count(&self) -> usize {
        self.success_count
    }
}
impl Clone for FaceProcessingResult {
    fn clone(&self) -> Self {
        FaceProcessingResult {
            total_count: self.total_count,
            success_count: self.success_count,
            failure_count: self.failure_count,
            failure_faces: self.failure_faces.clone(),
            missed_count: self.missed_count,
            missed_faces: self.missed_faces.clone(),
            context: self.context.clone(),
        }
    }
}
/// ProgressReporter enum to report the progress of the training or recognition operation
pub enum ProgressReporter<T>
where
    T: ProcessProgress + Clone + std::marker::Sync + std::marker::Send + 'static,
{
    /// Increase the progress fill by the given value
    Increase(u64),
    /// Increase the progress length by the given value
    IncreaseLength(u64),
    /// Set the progress message
    Message(String),
    /// Send the partial result of a single api call
    PartialStructedMessage(T),
    /// Send the accumulated result of the progress
    AccumulatedStructedMessage(T),
    /// Finish the progress with the given message
    FinishWithMessage(String),
}

#[derive(Deserialize, Debug, Clone)]
pub struct Subject {
    pub name: String,
    #[allow(unused)]
    pub similarity: f64,
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
    #[clap(long, env = "DATASET_PATH")]
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

    /// error configuration options
    #[clap(flatten)]
    pub error_configuration: ErrorConfiguration,
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

// error configuration options
#[derive(Debug, clap::Parser, Clone)]
#[clap(name = "error-options")]
pub struct ErrorConfiguration {
    /// Optional path to save the output of the failure and missing recognition
    /// When set, it will save the output to this path
    #[clap(long, env = "OUTPUT_DIR", default_value = None)]
    pub output_dir: Option<String>,

    /// Error behavior mode - should the process copy, move or ignore the error files
    /// The default value is Ignore
    /// Possible values are: Copy, Move, Ignore
    #[clap(long, env = "ERROR_BEHAVIOR", default_value = "ignore")]
    pub error_behavior: ErrorBehavior,

    #[clap(long, env = "POST_RECOGNIZE_STRATEGY", default_value = "MaxSimilarity")]
    pub post_recognize_strategy: PostRecognizeStrategy,

    /// The threshold to use when the PostRecognizeStrategy is AboveThreshold
    #[clap(long, env = "ABOVE_THRESHOLD", default_value = "0.95")]
    pub above_threshold: Option<f64>,
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Copy)]
pub enum ErrorBehavior {
    Copy,
    Move,
    Ignore,
}

/// Post recognize options, it work together with the ErrorBehavior
/// When the error behavior is set to Copy or Move, the PostRecognizeOptions will be used to determine the behavior
/// So, if the error behavior is set to Copy or Move, and the PostRecognizeOptions is set to MaxSimilarity,
/// for each recognize file we will copy or move the file to the folder with the subject with the maximum similarity (one to one strategy)
/// If the PostRecognizeOptions is set to AboveThreshold, we will copy or move the file to the folder with the subject with similarity above the threshold, so its possible to get multiple files per original one recognized file
/// Example: we have file a.jpg in the folder Magic Johnson and the recognize process did not found it,
/// but found 4 possible subjects:
///     Michael Jordan with similarity 0.91,
///     Larry bird with similarity 0.6,
///     James worthy with similarity 0.93
///     Lebron James with similarity 0.85
/// If the PostRecognizeOptions is set to MaxSimilarity, the file will be copied or moved to the folder James Worthy (the subject with the maximum similarity)
/// If the PostRecognizeOptions is set to AboveThreshold(0.9), the file will be copied or moved to the folders James Worthy and Michael Jordan
/// On all cases, it will also create an empty file with the original file name and the extension .original_name
/// so, if the original file is a.jpg, the empty file will be a.jpg.original_name
/// The default value is MaxSimilarity
#[derive(ValueEnum, Clone, Debug, PartialEq, Copy)]
pub enum PostRecognizeStrategy {
    /// only copy or move the file to the output folder and keep the original file name
    KeepAsIs,
    /// use the subject with the maximum similarity
    MaxSimilarity,
    /// use all subjects with similarity above the threshold
    AboveThreshold,
}

pub async fn process_files<T, F, Fut>(
    config: &Configuration,
    tx: Sender<ProgressReporter<T>>,
    api_action: F,
) -> anyhow::Result<()>
where
    F: Fn(String, Vec<PathBuf>, Sender<ProgressReporter<T>>) -> Fut + Send + Sync,
    Fut: Future<Output = anyhow::Result<()>> + Send,
    T: Clone + ProcessProgress + std::marker::Sync + std::marker::Send + 'static,
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

        // increase the progress length by the number of files in the group, ignoring directories or errors
        let files_count = group
            .iter()
            .filter(|path| path.is_ok() && path.as_ref().unwrap().is_file())
            .count();
        tx.send(ProgressReporter::IncreaseLength(files_count as u64))
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
                api_action(name.clone(), files_content.clone(), tx.clone()).await?;
                files_content.clear();
                total_size = 0;
            }

            total_size += file_len;
            files_content.push(path_buf);
        }

        if !files_content.is_empty() {
            api_action(name, files_content, tx.clone()).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {}
