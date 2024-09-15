mod compreface_client;
use compreface_client::CompreFaceClient;
use shared_api::{process_files, FaceProcessingResult, Recognizer, Trainer};
use shared_api::{Configuration, ProgressReporter};
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::Mutex;

pub async fn train(
    config: &Configuration,
    progress_reporter_tx: Sender<ProgressReporter<FaceProcessingResult>>,
) -> anyhow::Result<FaceProcessingResult> {
    let api_client = Arc::new(CompreFaceClient::new(config.compreface.clone().unwrap()));
    let state = Arc::new(Mutex::new(FaceProcessingResult::with_context(
        config.dataset_path.to_string(),
    )));
    let state_result = state.clone();
    let process_progress_reporter_tx = progress_reporter_tx.clone();
    let api_progress_reporter_tx = progress_reporter_tx.clone();

    process_files(
        config,
        process_progress_reporter_tx,
        move |name: String, files, process_progress_reporter_tx| {
            let api_client = Arc::clone(&api_client);
            let cloned_result = state.clone();
            let cloned_tx = api_progress_reporter_tx.clone();
            async move {
                let partial_result = api_client
                    .send_to_train(&name, files, process_progress_reporter_tx)
                    .await?;
                // accumulate the result
                let mut guard = cloned_result.lock().await;
                guard.add(partial_result);
                let report: FaceProcessingResult = guard.clone();
                cloned_tx
                    .send(ProgressReporter::AccumulatedStructedMessage(report))
                    .await?;

                Ok(())
            }
        },
    )
    .await?;
    let state_result = state_result.lock().await.clone();
    Ok(state_result)
}

pub async fn recognize(
    config: &Configuration,
    progress_reporter_tx: Sender<ProgressReporter<FaceProcessingResult>>,
) -> anyhow::Result<FaceProcessingResult> {
    let api_client = Arc::new(CompreFaceClient::new(config.compreface.clone().unwrap()));
    let state = Arc::new(Mutex::new(FaceProcessingResult::with_context(
        config.dataset_path.to_string(),
    )));
    let state_result = state.clone();
    let process_progress_reporter_tx = progress_reporter_tx.clone();
    let api_progress_reporter_tx = progress_reporter_tx.clone();
    process_files(
        config,
        process_progress_reporter_tx,
        move |name: String, files, process_progress_reporter_tx| {
            let api_client = Arc::clone(&api_client);
            let cloned_result = state.clone();
            let cloned_tx = api_progress_reporter_tx.clone();
            async move {
                let partial_result = api_client
                    .recognize(&name, files, process_progress_reporter_tx)
                    .await?;
                // accumulate the result
                let mut guard = cloned_result.lock().await;
                guard.add(partial_result);
                let report: FaceProcessingResult = guard.clone();
                cloned_tx
                    .send(ProgressReporter::AccumulatedStructedMessage(report))
                    .await?;

                Ok(())
            }
        },
    )
    .await?;
    let state_result = state_result.lock().await.clone();
    Ok(state_result)
}
