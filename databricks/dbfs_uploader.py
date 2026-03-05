import os
import configparser
import requests
import base64


CONFIG_PATH = "databricks/databricks.cfg"
LOCAL_DATA_DIR = "data"
DBFS_BASE_PATH = "dbfs:/Workspace/Users/uzochukwuekezie5@gmail.com/energy_data"



class DatabricksConfig:
    def __init__(self, config_path=CONFIG_PATH):
        parser = configparser.ConfigParser()
        parser.read(config_path)

        self.host = parser["DEFAULT"]["host"].rstrip("/")
        self.token = parser["DEFAULT"]["token"]

        if not self.host or not self.token:
            raise ValueError("Databricks host or token missing in config file")


class DBFSUploader:
    def __init__(self, config: DatabricksConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.token}"
        }

    def _api(self, endpoint: str) -> str:
        return f"{self.config.host}/api/2.0/dbfs/{endpoint}"

    def mkdirs(self, dbfs_path: str):
        requests.post(
            self._api("mkdirs"),
            headers=self.headers,
            json={"path": dbfs_path}
        )

    def upload_file(self, local_path: str, dbfs_path: str):
        with open(local_path, "rb") as f:
            data = f.read()

        encoded = base64.b64encode(data).decode("utf-8")

        resp = requests.post(
            self._api("put"),
            headers=self.headers,
            json={
                "path": dbfs_path,
                "contents": encoded,
                "overwrite": True
            }
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"DBFS put failed ({resp.status_code}): {resp.text}"
            )


    def file_exists(self, dbfs_path: str) -> bool:
        resp = requests.get(
            self._api("get-status"),
            headers=self.headers,
            params={"path": dbfs_path}
        )
        return resp.status_code == 200

    def upload_all_parquets(self):
        self.mkdirs(DBFS_BASE_PATH)

        for file in os.listdir(LOCAL_DATA_DIR):
            if file.endswith(".parquet"):
                local_file = os.path.join(LOCAL_DATA_DIR, file)
                dbfs_file = f"{DBFS_BASE_PATH}/{file}"

                print(f"Uploading {file}")
                self.upload_file(local_file, dbfs_file)

                if not self.file_exists(dbfs_file):
                    raise RuntimeError(f"Upload failed for {file}")

                print(f"Verified {file} in DBFS")

        print("All parquet files uploaded and verified")


def main():
    print("=== Starting DBFS Upload ===")

    config = DatabricksConfig()
    uploader = DBFSUploader(config)

    uploader.upload_all_parquets()

    print("=== DBFS Upload Completed ===")


if __name__ == "__main__":
    main()
