mod compreface_client;
use compreface_client::CompreFaceClient;
use shared_api::{process_files, RecognizeResult, Recognizer, TrainResult, Trainer};
use shared_api::{Configuration, ProgressReporter};
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::Mutex;

pub async fn train(
    config: &Configuration,
    tx: Sender<ProgressReporter<TrainResult>>,
) -> anyhow::Result<()> {
    let api_client = Arc::new(CompreFaceClient::new(config.compreface.clone().unwrap()));

    process_files(config, tx, move |name: String, files| {
        let api_client = Arc::clone(&api_client);
        async move {
            api_client.send_to_train(&name, files).await?;
            Ok(())
        }
    })
    .await
}

pub async fn recognize(
    config: &Configuration,
    progress_reporter_tx: Sender<ProgressReporter<RecognizeResult>>,
) -> anyhow::Result<RecognizeResult> {
    let api_client = Arc::new(CompreFaceClient::new(config.compreface.clone().unwrap()));
    let state = Arc::new(Mutex::new(RecognizeResult::with_context(
        config.dataset_path.to_string(),
    )));
    let state_result = state.clone();
    let process_progress_reporter_tx = progress_reporter_tx.clone();
    let api_progress_reporter_tx = progress_reporter_tx.clone();
    process_files(
        config,
        process_progress_reporter_tx,
        move |name: String, files| {
            let api_client = Arc::clone(&api_client);
            let cloned_result = state.clone();
            let cloned_tx = api_progress_reporter_tx.clone();
            async move {
                let partial_result = api_client.recognize(&name, files).await?;
                cloned_tx
                    .send(ProgressReporter::StructedMessage(partial_result.clone()))
                    .await?;
                // accumulate the result
                let mut guard = cloned_result.lock().await;
                guard.add(partial_result);
                // let report: RecognizeResult = guard.clone();
                // cloned_tx
                //     .send(ProgressReporter::StructedMessage(report))
                //     .await?;

                Ok(())
            }
        },
    )
    .await?;
    let state_result = state_result.lock().await.clone();
    progress_reporter_tx
        .send(ProgressReporter::StructedMessage(state_result.clone()))
        .await?;
    Ok(state_result)
}
