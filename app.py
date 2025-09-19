import boto3
import pandas as pd
import pathlib
from dotenv import load_dotenv

BUCKET_NAME = "dane-modul9"

load_dotenv()

s3 = boto3.client(
    "s3",
)

wroclaw_2023_df = pd.read_csv(f"s3://{BUCKET_NAME}/dane-zadanie_modul9/halfmarathon_wroclaw_2023__final.csv", sep=";")
wroclaw_2024_df = pd.read_csv(f"s3://{BUCKET_NAME}/dane-zadanie_modul9/halfmarathon_wroclaw_2024__final.csv", sep=";")

