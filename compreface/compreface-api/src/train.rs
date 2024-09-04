use crate::CompreFaceClient;
use async_trait::async_trait;
use futures::StreamExt;
use shared_api::utils;
use shared_api::Configuration;
use shared_api::ProgressReporter;
use shared_api::TrainLogic;
use shared_api::Trainer;
use std::path::PathBuf;
use stream_utils::buffer_until_condition::BufferUntilCondition;
use stream_utils::folder_scanner::RecursiveFileStream;
use tokio::fs;
use tokio::sync::mpsc::Sender;
use tracing::debug;

pub struct ComprefaceTrainLogic;

impl ComprefaceTrainLogic {
    pub fn new() -> Self {
        ComprefaceTrainLogic {}
    }
}

#[async_trait]
impl TrainLogic for ComprefaceTrainLogic {
    async fn train(config: &Configuration, tx: Sender<ProgressReporter>) -> anyhow::Result<()> {
        tx.send(ProgressReporter::Message(format!(
            "Start training directory: {}",
            &config.dataset_path
        )))
        .await?;
        let files = RecursiveFileStream::new(&config.dataset_path);
        // run the stream and group files into directories
        let mut files_groups =
            BufferUntilCondition::new(files, |path| path.as_ref().unwrap().is_dir());

        let api_client = CompreFaceClient::new(config.compreface.clone().unwrap());

        /* each group represent multiple files in the same directory,
        we should take the name from the directory, and each file content to an array of bytes arrays
        and send it to the double-take service
        we can use combinators, first to filter an map, and then take the first file and extract the directory name from it
        Then, we can map the rest of the files to extract the content of each file
        But I'm afraid that it cause to repeat iterating over the files, so I will use a for loop instead
         */
        while let Some(group) = files_groups.next().await {
            let name = match config.override_trained_name {
                Some(ref name) => name,
                None => &utils::get_directory_name(&group)?,
            };

            debug!(
                "group: {:?}, name: {}",
                group.first().unwrap().as_ref().unwrap().display(),
                name
            );

            tx.send(ProgressReporter::IncreaseLength(group.len() as u64))
                .await?;
            tx.send(ProgressReporter::Message(format!(
                "training directory: {}",
                name
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
                    debug!(
                        "file: {} is not of image type, skipping",
                        path_buf.display()
                    );
                    continue;
                }

                let file_len = fs::metadata(path_buf.clone()).await?.len();
                if total_size + file_len > config.max_request_size {
                    debug!(
                    "file {:?} size: {} + total size: {} > max_request_size {}, it will send the previous files without the current which will be sent in the next request",
                    &path_buf, file_len, total_size, config.max_request_size
                );
                    let current_items = files_content.len() as u64;

                    match api_client.send_to_train(&name, files_content).await {
                        Ok(_) => {
                            debug!(
                                "files sent successfully to train the model under the name: {}",
                                name
                            );
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                    tx.send(ProgressReporter::Increase(current_items)).await?;
                    files_content = Vec::new();
                    total_size = 0;
                }

                total_size += file_len;
                debug!(
                "file {:?} size: {} added to the list of files that will send later , total size: {}",
                &path_buf, file_len, total_size);
                files_content.push(path_buf);
            }
            if !files_content.is_empty() {
                let current_items = files_content.len() as u64;
                match api_client.send_to_train(&name, files_content).await {
                    Ok(_) => {
                        debug!(
                            "files sent successfully to train the model under the name: {}",
                            name
                        );
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
                tx.send(ProgressReporter::Increase(current_items)).await?;
            }
        }
        Ok(())
    }
}
