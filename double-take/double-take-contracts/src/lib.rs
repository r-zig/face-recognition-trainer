#[derive(Debug, clap::Parser, Clone)]
pub struct DoubleTakeConfig {
    /// The URL of double-take address
    #[clap(
        long,
        env = "DOUBLE_TAKE_URL",
        // under k8s
        // default_value = "http://my-double-take.ha:3000"
        default_value = "http://localhost:3000",
        help = "Double-take API URL"
    )]
    pub doubletake_url: String,
}
