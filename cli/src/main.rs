use compreface_api::train;
use dotenv::dotenv;
use indicatif::{MultiProgress, ProgressBar};
use shared_api::TrainLogic;
use shared_api::{ClientMode, Configuration, ProgressReporter};
use tokio::task;
use tracing::{debug, error, info};
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
    let progress_bar = multi_progress_bar.add(ProgressBar::new(0));

    if let Ok(style) = indicatif::ProgressStyle::default_spinner()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
        .template("{spinner:.green} [{elapsed_precise}] process folder 💡: {msg:.yellow.bold} [{wide_bar:.green/green}] {pos}/{len} files (p/s: {per_sec})")
    {
        progress_bar.set_style(style);
    }
    progress_bar.set_message("starting");

    // create rx,tx pair that will be used to send the progress report from the internal logic to the progress bar
    let (tx_progress, mut rx_progress) = tokio::sync::mpsc::channel(2);

    // spawn the async task that will run the logic, let the ui get the updates while the long process is running
    let long_task = task::spawn(async move {
        match config.client_mode {
            ClientMode::Train => {
                train::ComprefaceTrainLogic::train(&config, tx_progress).await?;
            }
            ClientMode::Recognize => {
                todo!();
                // recognize(files_groups, &config, tx).await?;
            }
        };
        anyhow::Result::<()>::Ok(()) // Explicit Ok return with specific type
    });

    // wait for notifications on the rx channel
    let reporting_task = task::spawn(async move {
        while let Some(progress_report) = rx_progress.recv().await {
            update_progress(progress_report, &progress_bar);
        }
        anyhow::Result::<()>::Ok(()) // Explicit Ok return with specific type
    });

    // wait for the long task to finish and then finish the progress bar
    match tokio::try_join!(long_task, reporting_task) {
        Ok(_) => debug!("Both tasks succeeded"),
        Err(e) => {
            error!("One of the tasks failed: {}", e);
            eprintln!("One of the tasks failed: {}", e);
        }
    }
    println!("finished");

    Ok(())
}

fn update_progress(progress_report: ProgressReporter, progress_bar: &ProgressBar) {
    match progress_report {
        ProgressReporter::Increase(len) => progress_bar.inc(len),
        ProgressReporter::IncreaseLength(len) => progress_bar.inc_length(len),
        ProgressReporter::Message(message) => progress_bar.set_message(message),
        ProgressReporter::FinishWithMessage(message) => progress_bar.finish_with_message(message),
    }
}
