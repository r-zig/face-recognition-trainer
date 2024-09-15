use std::path::PathBuf;

use anyhow::bail;
use compreface_api::{recognize, train};
use dotenv::dotenv;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use shared_api::{
    ClientMode, Configuration, ErrorBehavior, ErrorConfiguration, FaceProcessingResult,
    FailureFace, PostRecognizeStrategy, ProcessProgress, ProgressReporter,
};
use tokio::{
    fs::File,
    task::{self, JoinHandle},
};
use tracing::{debug, error, info, warn};
use tracing_bunyan_formatter::{BunyanFormattingLayer, JsonStorageLayer};
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Attempt to load the .env file, ignoring errors if the file doesn't exist
    dotenv().ok();
    let app_name = concat!(env!("CARGO_PKG_NAME"), "-", env!("CARGO_PKG_VERSION")).to_string();
    let (non_blocking_writer, _guard) = tracing_appender::non_blocking(std::io::stdout());
    let bunyan_formatting_layer = BunyanFormattingLayer::new(app_name.clone(), non_blocking_writer);
    let subscriber = Registry::default()
        .with(EnvFilter::from_default_env())
        .with(JsonStorageLayer)
        .with(bunyan_formatting_layer);
    tracing::subscriber::set_global_default(subscriber).unwrap();
    info!(app_name, "starting");

    let config = Configuration::get()?;

    let multi_progress_bar = MultiProgress::new();
    // represents the total files & folders progress bar
    let total_progress_bar = multi_progress_bar.add(ProgressBar::new(0));

    // represents the accumulated result of the process
    let accumulated_progress_bar = multi_progress_bar.add(ProgressBar::new(0));

    if let Ok(style) = indicatif::ProgressStyle::default_spinner()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
        .template("{spinner:.green} [{elapsed_precise}] process folder 💡: {msg:.yellow.bold} [{wide_bar:.green/green}] {pos}/{len} files (p/s: {per_sec})")
    {
        total_progress_bar.set_style(style);
    }

    accumulated_progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("Success rate: [{bar:40.cyan/blue}] {percent}% {pos}/{len} succeeded")
            .unwrap(),
    );

    total_progress_bar.set_message("starting");

    // create rx,tx pair that will be used to send the progress report from the internal logic to the progress bar
    let (tx_train_progress, mut rx_train_progress) = tokio::sync::mpsc::channel(2);
    let (tx_recognize_progress, mut rx_recognize_progress) = tokio::sync::mpsc::channel(2);

    let client_mode = config.client_mode.clone();
    // spawn the async task that will run the logic, let the ui get the updates while the long process is running
    let long_task = task::spawn(async move {
        let result = match config.client_mode {
            ClientMode::Train => train(&config, tx_train_progress.clone()).await?,
            ClientMode::Recognize => recognize(&config, tx_recognize_progress.clone()).await?,
        };
        tx_recognize_progress
            .send(ProgressReporter::AccumulatedStructedMessage(result.clone()))
            .await?;
        tx_recognize_progress
            .send(ProgressReporter::FinishWithMessage(format!(
                "Finish: {}",
                result
            )))
            .await?;
        Ok::<_, anyhow::Error>(())
    });

    // wait for notifications on the rx channel
    let reporting_task = task::spawn(async move {
        match client_mode {
            ClientMode::Train => {
                while let Some(progress_report) = rx_train_progress.recv().await {
                    on_progress::<FaceProcessingResult>(
                        progress_report,
                        &total_progress_bar,
                        &accumulated_progress_bar,
                    );
                }
            }
            ClientMode::Recognize => {
                while let Some(progress_report) = rx_recognize_progress.recv().await {
                    on_progress::<FaceProcessingResult>(
                        progress_report,
                        &total_progress_bar,
                        &accumulated_progress_bar,
                    );
                }
            }
        };

        anyhow::Result::<()>::Ok(()) // Explicit Ok return with specific type
    });

    match tokio::try_join!(flatten(long_task), flatten(reporting_task)) {
        Ok(_) => debug!("Both tasks succeeded"),
        Err(e) => {
            error!("One of the tasks failed: {}", e);
            eprintln!("One of the tasks failed: {}", e);
        }
    }
    println!("finished");

    Ok(())
}

async fn write_failures(
    config: &ErrorConfiguration,
    result: FaceProcessingResult,
) -> Result<(), anyhow::Error> {
    if config.error_behavior == ErrorBehavior::Ignore {
        return Ok(());
    }

    if result.failure_count > 0 {
        write_all_failure_faces(config, result.failure_faces).await?;
    }
    if result.missed_count > 0 {
        write_all_missing_faces(config, result.missed_faces).await?;
    }
    Ok(())
}

async fn write_all_missing_faces(
    config: &ErrorConfiguration,
    files: Vec<std::path::PathBuf>,
) -> Result<(), anyhow::Error> {
    let sub_folder = PathBuf::from(config.output_dir.as_ref().unwrap()).join("missed_faces");
    tokio::fs::create_dir_all(&sub_folder).await?;
    for path in files {
        let person_folder = sub_folder.join(path.parent().unwrap().file_stem().unwrap());
        tokio::fs::create_dir_all(&person_folder).await?;
        let file_name = path.file_name().unwrap().to_str().unwrap();
        let new_path = person_folder.join(file_name);
        match config.error_behavior {
            ErrorBehavior::Copy => {
                tokio::fs::copy(&path, &new_path).await?;
            }
            ErrorBehavior::Move => tokio::fs::rename(&path, &new_path).await?,
            ErrorBehavior::Ignore => {
                warn!(
                    "Ignoring the error behavior, so the file: {} will not be copy or moved",
                    path.display()
                );
            }
        }
    }
    Ok(())
}

async fn write_all_failure_faces(
    config: &ErrorConfiguration,
    faces: Vec<FailureFace>,
) -> Result<(), anyhow::Error> {
    let sub_folder = PathBuf::from(config.output_dir.as_ref().unwrap()).join("failure_faces");
    tokio::fs::create_dir_all(&sub_folder).await?;
    for failure_face in faces {
        let source_path = match failure_face {
            FailureFace::Train(ref path) => path.clone(),
            FailureFace::Recognize(ref m) => m.path.clone(),
        };

        // create the folder for the person
        let person_folder = sub_folder.join(source_path.parent().unwrap().file_stem().unwrap());
        tokio::fs::create_dir_all(&person_folder).await?;
        let file_name = source_path.file_name().unwrap().to_str().unwrap();
        // create the metadata files in the folder
        File::create(person_folder.join(file_name).join(".original_name")).await?;

        // act based on the strategy to save the file
        let target_files: Vec<PathBuf> = match failure_face {
            FailureFace::Train(_) => {
                // we should save the file as is
                vec![person_folder.join(file_name)]
            }
            FailureFace::Recognize(m) => {
                // we should save the file with the recognize suffix
                match config.post_recognize_strategy {
                    PostRecognizeStrategy::KeepAsIs => vec![person_folder.join(file_name)],
                    PostRecognizeStrategy::MaxSimilarity => {
                        if let Some(subject) = m
                            .subjects
                            .iter()
                            .max_by_key(|item| (item.similarity * 1000.0) as u32)
                        {
                            vec![person_folder.join(subject.name.clone())]
                        } else {
                            bail!("Failed to find the max similarity subject for the file: {}, # of subjects: {}",
                                source_path.display(),
                                m.subjects.len());
                        }
                    }
                    PostRecognizeStrategy::AboveThreshold => vec![m
                        .subjects
                        .iter()
                        .filter(|item| item.similarity > config.above_threshold.unwrap())
                        .map(|item| person_folder.join(item.name.clone()))
                        .collect()],
                }
            }
        };

        // move the file if there is only one target file
        if target_files.len() == 1 && config.error_behavior == ErrorBehavior::Move {
            tokio::fs::rename(&source_path, &target_files.first().unwrap()).await?;
        } else {
            for new_path in target_files {
                tokio::fs::copy(&source_path, &new_path).await?;
            }

            // should remove the source file if the error behavior is move - after completing the copy
            if config.error_behavior == ErrorBehavior::Move {
                tokio::fs::remove_file(&source_path).await?;
            }
        }
    }
    Ok(())
}

async fn flatten<T>(handle: JoinHandle<Result<T, anyhow::Error>>) -> Result<T, anyhow::Error> {
    match handle.await {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(err)) => Err(err),
        Err(err) => Err(err.into()),
    }
}

fn on_progress<T>(
    progress_report: ProgressReporter<T>,
    total_progress_bar: &ProgressBar,
    accumulated_progress_bar: &ProgressBar,
) where
    T: core::fmt::Display
        + ProcessProgress
        + Clone
        + std::marker::Sync
        + std::marker::Send
        + 'static,
{
    match progress_report {
        ProgressReporter::Increase(len) => {
            total_progress_bar.inc(len);
        }
        ProgressReporter::IncreaseLength(len) => total_progress_bar.inc_length(len),
        ProgressReporter::Message(message) => {
            total_progress_bar.set_message(message);
        }
        ProgressReporter::FinishWithMessage(message) => {
            total_progress_bar.finish_with_message(message)
        }
        ProgressReporter::PartialStructedMessage(message) => {
            // write the missing and failures files to the file
            // write_failures(&config.error_configuration, message.)
            // .await.map_err(|e| {
            //     warn!("Failed to write the missing and failures files, but the process finished. error: {}", e);
            // });
        }
        ProgressReporter::AccumulatedStructedMessage(message) => {
            accumulated_progress_bar.set_length(message.get_total_count() as u64);
            accumulated_progress_bar.set_position(message.get_success_count() as u64);
            accumulated_progress_bar.abandon();
        }
    }
}
