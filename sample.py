import requests

url = "https://nl-api.onrender.com/summarization/"
data = {
    "text": "Unit economics, a critical component of financial modeling, extends beyond basic revenue and cost analysis per unit. It delves into the intricate interplay of variables like customer acquisition cost (CAC), customer lifetime value (CLTV), and churn rate, often incorporating cohort analysis to understand how these metrics evolve over time. Furthermore, it necessitates a nuanced understanding of fixed versus variable costs, the impact of scaling on these costs, and the potential for network effects or economies of scale to influence profitability. In subscription-based models, for instance, unit economics may involve analyzing the impact of different pricing tiers, freemium conversions, and the long-term value of recurring revenue streams. A comprehensive unit economics model might also consider factors like customer referrals, upselling and cross-selling opportunities, and the impact of seasonality or external market forces on customer behavior. Ultimately, a robust unit economics framework enables businesses to not only assess current profitability but also forecast future performance under various scenarios, informing strategic decisions related to growth, investment, and operational optimization.",
    "max_length": 1000,
    "min_length": 50
}
response = requests.post(url, json=data)
print(response.json())
