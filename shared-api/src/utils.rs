use anyhow::anyhow;
use std::path::{Path, PathBuf};

use anyhow::bail;
use tracing::debug;

pub fn is_image(path: &Path) -> bool {
    let extension = path.extension().unwrap_or_default().to_string_lossy();
    extension == "jpg" || extension == "jpeg" || extension == "png"
}

pub fn get_directory_name(group: &[Result<PathBuf, std::io::Error>]) -> anyhow::Result<String> {
    let first_file = group.first().ok_or(anyhow!("empty group"))?;
    let path_buf = match first_file {
        Ok(path) => path.clone(),
        Err(e) => bail!("error: {}", e),
    };

    debug!("group: {:?}", path_buf);

    if path_buf.is_dir() {
        let name = path_buf
            .file_stem()
            .ok_or(anyhow!("folder name not found on path: {:?}", path_buf))?
            .to_string_lossy()
            .into_owned();
        Ok(name)
    } else {
        let name = path_buf
            .parent()
            .ok_or(anyhow!("parent folder not found on path: {:?}", path_buf))?
            .file_stem()
            .ok_or(anyhow!("folder name not found on path: {:?}", path_buf))?
            .to_string_lossy()
            .into_owned();
        Ok(name)
    }
}
#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_is_image_with_jpg_extension() {
        let path = Path::new("image.jpg");
        assert_eq!(is_image(&path), true);
    }

    #[test]
    fn test_is_image_with_jpeg_extension() {
        let path = Path::new("image.jpeg");
        assert_eq!(is_image(&path), true);
    }

    #[test]
    fn test_is_image_with_png_extension() {
        let path = Path::new("image.png");
        assert_eq!(is_image(&path), true);
    }

    #[test]
    fn test_is_image_with_invalid_extension() {
        let path = Path::new("image.txt");
        assert_eq!(is_image(&path), false);
    }

    #[test]
    fn test_get_directory_name_with_valid_directory() {
        let group = vec![Ok(PathBuf::from("/path/to/directory/image.jpg"))];
        assert_eq!(get_directory_name(&group).unwrap(), "directory".to_string());
    }

    #[test]
    fn test_get_directory_name_with_valid_directory_with_spaces() {
        let dir = tempdir().unwrap();
        let expected_dir = "sub dir";
        let subdir = dir.path().join(expected_dir);
        fs::create_dir(&subdir).unwrap();
        let group = vec![
            Ok(PathBuf::from(&subdir)),
            Ok(PathBuf::from(subdir).join("image.jpg")),
        ];
        assert_eq!(get_directory_name(&group).unwrap(), expected_dir);
    }

    #[test]
    fn test_get_directory_name_with_many_files_and_one_directory_entries() {
        let group:Vec<Result<PathBuf, std::io::Error>> = vec![Ok(PathBuf::from("/home/ron/Documents/smart-home/faces-train/known/aaa")),
         Ok(PathBuf::from("/home/ron/Documents/smart-home/faces-train/known/aaa/33e778d2-3336-4566-9f52-6bc468b7eb32-1720362645708.jpg")),
         Ok(PathBuf::from("/home/ron/Documents/smart-home/faces-train/known/aaa/8574cc19-dc28-4f13-aab0-161de0e4e3e7-1723111077185.jpg"))];
        assert_eq!(get_directory_name(&group).unwrap(), "aaa");
    }

    #[test]
    fn test_get_directory_name_with_invalid_path() {
        let group = vec![Err(std::io::Error::new(std::io::ErrorKind::Other, "error"))];
        assert!(get_directory_name(&group).is_err());
    }

    #[test]
    fn test_get_directory_name_with_empty_group() {
        let group: Vec<Result<PathBuf, std::io::Error>> = vec![];
        assert_eq!(
            get_directory_name(&group).unwrap_err().to_string(),
            "empty group"
        );
    }
}
