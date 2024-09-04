use std::{path::PathBuf, time::Duration};

use anyhow::bail;
use async_trait::async_trait;
use compreface_contracts::CompreFaceConfig;
use mime_guess::MimeGuess;
use reqwest::{multipart::Part, Client};
use serde::Deserialize;
use shared_api::{RecognizeResult, Recognizer, Trainer};
use tokio::{fs, io::AsyncReadExt, time::sleep};
use tracing::{debug, error, warn};

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
    async fn send_to_train(&self, name: &str, files: Vec<PathBuf>) -> anyhow::Result<()> {
        // this is postman example: {{compreface_base_url}}/api/v1/recognition/faces?subject={{subject_name}}
        let url = format!(
            "{}/api/v1/recognition/faces?subject={}",
            self.config.compreface_url, name
        );

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
                .await?;

            match response.status().as_u16() {
                200 => {
                    debug!(
                        "Training: {} for file: {} response: {}",
                        name,
                        file_path.display(),
                        &response.text().await?
                    );
                }
                201 => {
                    debug!(
                        "Training: {} for file: {} response: {}",
                        name,
                        file_path.display(),
                        &response.text().await?
                    );
                }
                400 => {
                    error!("Failed to train file: {}, for name: {}, response.status: {}, response text: {}, but will continue with the other files",
                        file_path.display(),
                        name,
                        &response.status(),
                        &response.text().await?
                    );
                    // move the file to the error directory
                    let original_extension = file_path.extension().unwrap().to_str().unwrap();
                    let renamed_path = file_path.as_path().parent().unwrap().join(
                        file_path
                            .as_path()
                            .file_stem()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .to_owned()
                            + ".error."
                            + original_extension,
                    );

                    warn!(
                        "Renaming file: {} to: {}",
                        file_path.display(),
                        renamed_path.display()
                    );
                    fs::rename(file_path.as_path(), renamed_path).await?;
                    continue;
                }
                _ => {
                    bail!(format!(
                        "Failed to train file: {}, for name: {}, response.status: {}, response text: {}",
                        file_path.display(),
                        name,
                        &response.status(),
                        &response.text().await?
                    ));
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Recognizer for CompreFaceClient {
    async fn recognize(&self, name: &str, files: Vec<PathBuf>) -> anyhow::Result<RecognizeResult> {
        // this is postman example: {{compreface_base_url}}/api/v1/recognition/recognize
        let url = format!(
            "{}/api/v1/recognition/recognize",
            self.config.compreface_url
        );

        debug!("recognizing directory {} with {} files", name, files.len());
        let mut recognition_result = RecognizeResult::with_capacity(files.len());

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
                .await?
                .json::<RecognitionApiResponse>()
                .await?;
            if response
                .result
                .iter()
                .all(|r| r.subjects.iter().all(|s| s.subject != name))
            {
                recognition_result.failure_count += 1;
                recognition_result.unrecognized.push(file_path);
                continue;
            } else {
                recognition_result.success_count += 1;
            }
        }
        Ok(recognition_result)
    }
}

#[derive(Deserialize, Debug)]
struct RecognitionApiResponse {
    result: Vec<ResultItem>,
}

#[derive(Deserialize, Debug)]
struct ResultItem {
    r#box: DetectionBox,
    subjects: Vec<Subject>,
}

#[derive(Deserialize, Debug)]
struct DetectionBox {
    probability: f64,
    x_max: u32,
    y_max: u32,
    x_min: u32,
    y_min: u32,
}

#[derive(Deserialize, Debug)]
struct Subject {
    subject: String,
    similarity: f64,
}
