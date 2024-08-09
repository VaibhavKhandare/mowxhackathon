const url = "https://e553-182-73-85-174.ngrok-free.app/generate_ads"; // Replace with your Flask app URL
const currentURL = window.location.href; // Get current page URL

//Send a POST request with the current page URL
const response = await fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ url: currentURL })
});

if (response.ok) {
    const data = response.json();
    const paragraphIndices = data.paragraph_indices;
    const ads = data.ads;

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
    console.error("Failed to retrieve ad placements:", response.statusText);
}

// Call the function when the page loads
window.onload = ()=>{
	sendHTMLToBackendAndInjectAds();
};