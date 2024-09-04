mod compreface_client;
use compreface_client::CompreFaceClient;
use shared_api::{process_files, Trainer};
use shared_api::{Configuration, ProgressReporter};
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

pub async fn train(config: &Configuration, tx: Sender<ProgressReporter>) -> anyhow::Result<()> {
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
