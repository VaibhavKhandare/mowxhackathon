const analyzeUrl = "https://24c9-14-99-167-102.ngrok-free.app/analyze_content"; // Replace with your Flask app URL for content analysis
const generateAdsUrl = "https://24c9-14-99-167-102.ngrok-free.app/generate_ads_only"; // Replace with your Flask app URL for ad generation

async function fetchAndInjectAds() {
    const currentURL = window.location.href; // Get current page URL

    // Step 1: Analyze content and get segment info
    const analyzeResponse = await fetch(analyzeUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: currentURL })
    });

    if (analyzeResponse.ok) {
        const analyzeData = await analyzeResponse.json();
        const paragraphIndices = analyzeData.paragraph_indices;
        const segmentInfo = analyzeData.segment_info;

        // Step 2: Generate ads based on segment info
        const generateAdsResponse = await fetch(generateAdsUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ segment_info: segmentInfo })
        });

        if (generateAdsResponse.ok) {
            const adsData = await generateAdsResponse.json();
            const ads = adsData.ads;

            // Inject ads into the corresponding paragraphs
            const paragraphs = document.getElementsByTagName('p');

            paragraphIndices.forEach((index, i) => {
                if (index < paragraphs.length) {
                    const adHtml = ads[i];
                    const adElement = document.createElement('p');
                    adElement.innerHTML = adHtml;
                    paragraphs[index].insertAdjacentElement('afterend', adElement);
                }
            });

            console.log("Ads injected successfully!");
        } else {
            console.error("Failed to generate ads:", generateAdsResponse.statusText);
        }
    } else {
        console.error("Failed to analyze content:", analyzeResponse.statusText);
    }
}

// Call the function when the page loads
window.onload = fetchAndInjectAds;
