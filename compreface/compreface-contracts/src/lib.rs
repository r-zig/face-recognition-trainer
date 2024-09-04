#[derive(Debug, clap::Parser, Clone)]
pub struct CompreFaceConfig {
    #[clap(
        long,
        env = "COMPREFACE_URL",
        default_value = "http://localhost:8080",
        help = "CompreFace API URL"
    )]
    pub compreface_url: String,

    #[clap(long, env = "COMPREFACE_API_KEY", help = "CompreFace API key")]
    pub compreface_api_key: String,
}
