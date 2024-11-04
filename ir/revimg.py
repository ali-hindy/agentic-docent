import boto3
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import time

load_dotenv()

def upload_image_to_s3(file_path, bucket_name, object_name=None):
    """Uploads an image to the specified S3 bucket."""
    s3_client = boto3.client('s3')
    
    # If no object name is provided, use the file name
    if object_name is None:
        object_name = os.path.basename(file_path)
        
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Image successfully uploaded to S3 bucket '{bucket_name}' as '{object_name}'.")
        return f"s3://{bucket_name}/{object_name}"
    except Exception as e:
        print(f"Failed to upload image: {e}")
        return None
    
def google_reverse_image_search(image_path):
    """Performs a Google Reverse Image Search using Selenium."""
    # Initialize Chrome WebDriver (path to ChromeDriver must be set)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run headless for non-GUI environments
    driver = webdriver.Chrome(options=options)
    
    try:
        # Open Google Images
        driver.get("https://images.google.com")
        
        # Click on the camera icon to initiate reverse image search
        camera_button = driver.find_element(By.XPATH, '//div[@aria-label="Search by image"]')
        camera_button.click()
        
        # Upload the image
        upload_tab = driver.find_element(By.XPATH, '//a[contains(text(), "Upload an image")]')
        upload_tab.click()
        
        # Upload the image file
        upload_input = driver.find_element(By.XPATH, '//input[@type="file"]')
        upload_input.send_keys(image_path)
        
        # Allow time for upload and search to complete
        time.sleep(5)
        
        # Retrieve and print top search results
        results = driver.find_elements(By.XPATH, '//div[@class="yuRUbf"]/a')
        top_results = [result.get_attribute("href") for result in results[:5]]  # Get top 5 results
        return top_results
    
    except Exception as e:
        print(f"Failed to perform reverse image search: {e}")
        return None
    finally:
        driver.quit()
    
if __name__ == "__main__":
    bucket_name = "224v-docent"
    image_path = "../scrape/data/images/akkitham-narayanan_untitled-6.jpg"
    upload_image_to_s3(image_path, bucket_name)
    # Perform reverse image search
    results = google_reverse_image_search(image_path)
    print(results)
