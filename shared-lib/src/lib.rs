use shared_api::{Configuration, TrainLogic};

// pub fn get_train_logic(
//     config: &Configuration,
// ) -> Result<Box<dyn TrainLogic>, Box<dyn std::error::Error>> {
//     match config.client_type {
//         shared_api::ClientType::Compreface => Ok(Box::new(shared_api::ComprefaceTrainLogic::new())),
//         shared_api::ClientType::DoubleTake => todo!(),
//     }
// }
