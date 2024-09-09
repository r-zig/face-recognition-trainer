use std::path::Path;

use compreface_api::{recognize, train};
use dotenv::dotenv;
use indicatif::{MultiProgress, ProgressBar};
use shared_api::{ClientMode, Configuration, ProgressReporter, RecognizeResult, TrainResult};
use tokio::task::{self, JoinHandle};
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
    total_progress_bar.set_message("starting");

    // create rx,tx pair that will be used to send the progress report from the internal logic to the progress bar
    let (tx_train_progress, mut rx_train_progress) = tokio::sync::mpsc::channel(2);
    let (tx_recognize_progress, mut rx_recognize_progress) = tokio::sync::mpsc::channel(2);

    let client_mode = config.client_mode.clone();
    // spawn the async task that will run the logic, let the ui get the updates while the long process is running
    let long_task = task::spawn(async move {
        match config.client_mode {
            ClientMode::Train => {
                train(&config, tx_train_progress).await?;
            }
            ClientMode::Recognize => {
                let result = recognize(&config, tx_recognize_progress).await?;
                // write the missing and failures files to the file
                write_failures(&config, result).await.map_err(|e| {
                    warn!("Failed to write the missing and failures files, but the process finished. error: {}", e);
                    e
                })?;
            }
        };
        Ok::<_, anyhow::Error>(())
    });

    // wait for notifications on the rx channel
    let reporting_task = task::spawn(async move {
        match client_mode {
            ClientMode::Train => {
                while let Some(progress_report) = rx_train_progress.recv().await {
                    update_progress::<TrainResult>(
                        progress_report,
                        &total_progress_bar,
                        &accumulated_progress_bar,
                    );
                }
            }
            ClientMode::Recognize => {
                while let Some(progress_report) = rx_recognize_progress.recv().await {
                    update_progress::<RecognizeResult>(
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
    config: &Configuration,
    result: RecognizeResult,
) -> Result<(), anyhow::Error> {
    if let Some(output_dir) = &config.output_dir {
        let output_dir: &Path = Path::new(output_dir);
        if result.failure_count > 0 {
            let sub_folder = output_dir.join("failure_faces");
            write_all_faces(sub_folder, result.failure_faces).await?;
        }
        if result.missed_count > 0 {
            let sub_folder = output_dir.join("missed_faces");
            write_all_faces(sub_folder, result.missed_faces).await?;
        }
    }
    Ok(())
}

async fn write_all_faces(
    sub_folder: std::path::PathBuf,
    files: Vec<std::path::PathBuf>,
) -> Result<(), anyhow::Error> {
    tokio::fs::create_dir_all(&sub_folder).await?;
    for path in files {
        let person_folder = sub_folder.join(path.parent().unwrap().file_stem().unwrap());
        tokio::fs::create_dir_all(&person_folder).await?;
        let file_name = path.file_name().unwrap().to_str().unwrap();
        let new_path = person_folder.join(file_name);
        tokio::fs::copy(&path, &new_path).await?;
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

fn update_progress<T>(
    progress_report: ProgressReporter<T>,
    total_progress_bar: &ProgressBar,
    accumulated_progress_bar: &ProgressBar,
) where
    T: core::fmt::Display + Clone + std::marker::Sync + std::marker::Send + 'static,
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
        ProgressReporter::StructedMessage(message) => {
            total_progress_bar.set_message(message.to_string());
            accumulated_progress_bar.set_message(message.to_string());
            println!("** {}", message);
        }
    }
}
