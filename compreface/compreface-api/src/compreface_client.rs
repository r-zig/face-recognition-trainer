use std::path::PathBuf;

use async_trait::async_trait;
use compreface_contracts::CompreFaceConfig;
use mime_guess::MimeGuess;
use reqwest::{multipart::Part, Client};
use serde::Deserialize;
use shared_api::{
    FaceProcessingResult, FaceWithMetadata, FailureFace, ProgressReporter, Recognizer, Subject,
    Trainer,
};
use tokio::{fs, io::AsyncReadExt, sync::mpsc::Sender};
use tracing::{debug, error};

/// Comperface client supports handling communication with the Comperface API.
pub struct CompreFaceClient {
    client: Client,
    config: CompreFaceConfig,
}

impl CompreFaceClient {
    pub fn new(config: CompreFaceConfig) -> Self {
        let client = Client::new();
        CompreFaceClient { client, config }
    }
}

#[async_trait]
impl Trainer for CompreFaceClient {
    async fn send_to_train(
        &self,
        name: &str,
        files: Vec<PathBuf>,
        progress_reporter_tx: Sender<ProgressReporter<FaceProcessingResult>>,
    ) -> anyhow::Result<FaceProcessingResult> {
        // this is postman example: {{compreface_base_url}}/api/v1/recognition/faces?subject={{subject_name}}
        let url = format!(
            "{}/api/v1/recognition/faces?subject={}",
            self.config.compreface_url, name
        );

        let mut recognition_result = FaceProcessingResult::with_context(
            files
                .first()
                .unwrap()
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
        );

        recognition_result.total_count = files.len();
        debug!("training directory {} with {} files", name, files.len());
        for file_path in files {
            debug!("sending file: {:?}", file_path);

            let mime = MimeGuess::from_path(file_path.as_path()).first_or_octet_stream();
            let mut file = fs::File::open(file_path.as_path()).await?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).await?;

            let part = Part::bytes(buffer)
                .file_name(
                    file_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                )
                .mime_str(mime.as_ref())?;
            let form = reqwest::multipart::Form::new().part("file", part);

            let response = self
                .client
                .post(&url)
                .header("x-api-key", &self.config.compreface_api_key)
                .multipart(form)
                .send()
                .await;
            if let Err(e) = response {
                error!(
                    "Failed to train file: {} for name: {}: {}",
                    file_path.display(),
                    name,
                    e
                );
                recognition_result.missed_count += 1;
                recognition_result.missed_faces.push(file_path);
                continue;
            }
            let response = response.unwrap();
            progress_reporter_tx
                .send(ProgressReporter::Increase(1))
                .await?;
            match response.status().as_u16() {
                200 | 201 => {
                    recognition_result.success_count += 1;
                    debug!(
                        "Training: {} for file: {} response: {}",
                        name,
                        file_path.display(),
                        &response.text().await?
                    );
                }
                _ => {
                    error!("Failed to train file: {}, for name: {}, response.status: {}, response text: {}, but will continue with the other files",
                        file_path.display(),
                        name,
                        &response.status(),
                        &response.text().await?
                    );
                    recognition_result.failure_count += 1;
                    recognition_result
                        .failure_faces
                        .push(FailureFace::Train(file_path));
                    continue;
                }
            }
        }
        Ok(recognition_result)
    }
}

#[async_trait]
impl Recognizer for CompreFaceClient {
    async fn recognize(
        &self,
        name: &str,
        files: Vec<PathBuf>,
        progress_reporter_tx: Sender<ProgressReporter<FaceProcessingResult>>,
    ) -> anyhow::Result<FaceProcessingResult> {
        // this is postman example: {{compreface_base_url}}/api/v1/recognition/recognize
        let url = format!(
            "{}/api/v1/recognition/recognize",
            self.config.compreface_url
        );

        debug!("recognizing directory {} with {} files", name, files.len());
        let mut recognition_result = FaceProcessingResult::with_context(
            files
                .first()
                .unwrap()
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
        );

        for file_path in files {
            debug!("sending file: {:?}", file_path);
            recognition_result.total_count += 1;

            let mime = MimeGuess::from_path(file_path.as_path()).first_or_octet_stream();
            let mut file = fs::File::open(file_path.as_path()).await?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).await?;

            let part = Part::bytes(buffer)
                .file_name(
                    file_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                )
                .mime_str(mime.as_ref())?;
            let form = reqwest::multipart::Form::new().part("file", part);

            let response = match self
                .client
                .post(&url)
                .header("x-api-key", &self.config.compreface_api_key)
                .multipart(form)
                .send()
                .await
            {
                Ok(response) => response,
                Err(e) => {
                    error!(
                        "Failed to recognize file: {} for name: {}: {}",
                        file_path.display(),
                        name,
                        e
                    );
                    recognition_result.missed_count += 1;
                    recognition_result.missed_faces.push(file_path);
                    progress_reporter_tx
                        .send(ProgressReporter::Increase(1))
                        .await?;
                    continue;
                }
            };
            match response.json::<RecognitionApiResponse>().await {
                Ok(response) => {
                    if response
                        .result
                        .iter()
                        .any(|r| r.subjects.iter().any(|s| s.name == name))
                    {
                        recognition_result.success_count += 1;
                    } else {
                        recognition_result.failure_count += 1;
                        recognition_result
                            .failure_faces
                            .push(FailureFace::Recognize(FaceWithMetadata {
                                path: file_path,
                                subjects: response.get_subjects(),
                            }));
                    }
                }
                Err(e) => {
                    error!(
                        "Failed to parse JSON response for file: {} for name: {} Error: {}",
                        file_path.display(),
                        name,
                        e
                    );
                    recognition_result.missed_count += 1;
                    recognition_result.missed_faces.push(file_path);
                }
            }
            progress_reporter_tx
                .send(ProgressReporter::Increase(1))
                .await?;
        }
        Ok(recognition_result)
    }
}

#[derive(Deserialize, Debug)]
struct RecognitionApiResponse {
    result: Vec<ResultItem>,
}

impl RecognitionApiResponse {
    fn get_subjects(&self) -> Vec<Subject> {
        self.result
            .iter()
            .flat_map(|r| r.subjects.clone())
            .collect()
    }
}

#[derive(Deserialize, Debug)]
struct ResultItem {
    #[allow(unused)]
    r#box: DetectionBox,
    subjects: Vec<Subject>,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
struct DetectionBox {
    probability: f64,
    x_max: u32,
    y_max: u32,
    x_min: u32,
    y_min: u32,
}
