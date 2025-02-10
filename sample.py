import requests
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

url = "https://nl-api.onrender.com/summarization/"
data = {
    "text": "Unit economics, a critical component of financial modeling, extends beyond basic revenue and cost analysis per unit. It delves into the intricate interplay of variables like customer acquisition cost (CAC), customer lifetime value (CLTV), and churn rate, often incorporating cohort analysis to understand how these metrics evolve over time. Furthermore, it necessitates a nuanced understanding of fixed versus variable costs, the impact of scaling on these costs, and the potential for network effects or economies of scale to influence profitability. In subscription-based models, for instance, unit economics may involve analyzing the impact of different pricing tiers, freemium conversions, and the long-term value of recurring revenue streams. A comprehensive unit economics model might also consider factors like customer referrals, upselling and cross-selling opportunities, and the impact of seasonality or external market forces on customer behavior. Ultimately, a robust unit economics framework enables businesses to not only assess current profitability but also forecast future performance under various scenarios, informing strategic decisions related to growth, investment, and operational optimization.",
    "max_length": 1000,
    "min_length": 50
}

def make_request():
    max_retries = 5
    retry_delay = 15  # seconds
    timeout = 180  # 3 minutes timeout

    for attempt in range(max_retries):
        try:
            logger.info(f"\nAttempt {attempt + 1} of {max_retries}")
            logger.info("Sending request to summarization endpoint...")
            start_time = time.time()
            
            response = requests.post(
                url,
                json=data,
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            duration = time.time() - start_time
            logger.info(f"Request took {duration:.2f} seconds")

            response.raise_for_status()
            
            result = response.json()
            print("\n=== Summary Result ===")
            print("\nOriginal Text Length:", len(data["text"]))
            print("Summary Length:", len(result["summary"]))
            print("\nSummary:")
            print(result["summary"])
            print("\n===================")
            
            return True

        except requests.exceptions.Timeout:
            logger.error(f"Attempt {attempt + 1} failed: Request timed out")
        except requests.exceptions.ConnectionError:
            logger.error(f"Attempt {attempt + 1} failed: Connection error")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Attempt {attempt + 1} failed: HTTP {e.response.status_code} error")
            if e.response.status_code == 502:
                logger.info("Server is probably still loading the models...")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: Unexpected error: {str(e)}")
        
        if attempt < max_retries - 1:
            logger.info(f"Waiting {retry_delay} seconds before retrying...")
            time.sleep(retry_delay)
        else:
            logger.error("All attempts failed.")
            return False

if __name__ == "__main__":
    logger.info("Starting API test...")
    logger.info("Proceeding with summarization request...")
    make_request()