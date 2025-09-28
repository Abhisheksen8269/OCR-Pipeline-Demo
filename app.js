// --- Configuration ---
const API_ENDPOINT = '/extract-amounts';
const CONFIDENCE_THRESHOLDS = {
    HIGH: 0.90, // Green/Success zone
    MEDIUM: 0.75  // Amber/Warning zone
};

// --- Helper Functions ---

/**
 * Applies color styling to confidence badges based on score.
 * @param {number} confidence - The confidence score (0.00 to 1.00).
 * @returns {string} The CSS class for coloring.
 */
function getConfidenceClass(confidence) {
    if (confidence >= CONFIDENCE_THRESHOLDS.HIGH) {
        return 'conf-high'; // Green
    } else if (confidence >= CONFIDENCE_THRESHOLDS.MEDIUM) {
        return 'conf-medium'; // Amber/Yellow
    } else {
        return 'conf-low'; // Red
    }
}

/**
 * Renders the final classified amounts into a clean list for the Step 4 visualization.
 * @param {Array<Object>} amounts - List of final classified amounts.
 * @param {string} currency - The currency (e.g., 'USD').
 * @returns {string} HTML string of structured list items.
 */
function renderFinalAmountCards(amounts, currency) {
    if (!amounts || amounts.length === 0) {
        return '<p class="text-gray-500 text-center col-span-2">No primary amounts classified.</p>';
    }

    return amounts.map(item => {
        const value = `${currency} ${item.value.toLocaleString()}`;
        const typeDisplay = item.type.replace('_', ' ').toUpperCase();
        
        let itemClass = 'summary-item';
        let typeColor = 'text-gray-500';

        if (item.type === 'total_bill') {
            itemClass += ' summary-item-total';
            typeColor = 'text-blue-600';
        } else if (item.type === 'paid') {
            itemClass += ' summary-item-paid';
            typeColor = 'text-green-600';
        } else if (item.type === 'due') {
            itemClass += ' summary-item-due';
            typeColor = 'text-red-600';
        } else if (item.type === 'discount') {
            typeColor = 'text-amber-600';
        }

        return `
            <div class="${itemClass}">
                <div class="text-xs font-medium ${typeColor}">${typeDisplay}</div>
                <div class="amount">${value}</div>
                <div class="text-xs text-gray-500 mt-1 overflow-hidden whitespace-nowrap overflow-ellipsis" title="${item.source}">
                    Source: ${item.source.substring(7, 60)}
                </div>
            </div>
        `;
    }).join('');
}


/**
 * Applies JSON highlighting (used for pre tags).
 */
function highlightJson(json) {
    if (typeof json != 'string') {
         json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'json-number';
        if (/^"/.test(match)) {
            cls = /:$/.test(match) ? 'json-key' : 'json-string';
        } else if (/true|false/.test(match)) {
            cls = 'json-boolean';
        } else if (/null/.test(match)) {
            cls = 'json-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}


/**
 * Primary function to handle form submission and orchestrate the display.
 */
async function handleSubmit(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const loadingOverlay = document.getElementById('loading-overlay');
    const initialMsg = document.getElementById('initial-message');
    const errorMsg = document.getElementById('error-message');
    const resultsPanel = document.getElementById('results-panel');
    const submitButton = document.getElementById('submit-button');

    // 1. Setup UI for processing (Show loading, hide results/errors)
    submitButton.disabled = true;
    initialMsg.classList.add('hidden');
    errorMsg.classList.add('hidden');
    resultsPanel.classList.add('hidden');
    loadingOverlay.style.display = 'flex';
    document.getElementById('final-summary-list').innerHTML = ''; // Clear visual summary

    try {
        const response = await fetch('/extract-amounts', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        loadingOverlay.style.display = 'none';
        submitButton.disabled = false;

        if (response.ok && data.step_4_final_result && data.step_4_final_result.status === 'ok') {
            const steps = data;
            resultsPanel.classList.remove('hidden');

            // --- Confidence Display Helper ---
            const updateConfidenceBadge = (id, score) => {
                const badge = document.getElementById(id);
                badge.textContent = score.toFixed(2);
                badge.className = `conf-badge ${getConfidenceClass(score)}`;
            };

            // 2. Populate Step 1 (Raw Extraction)
            updateConfidenceBadge('conf-1', steps.step_1_raw_extraction.confidence);
            document.getElementById('result-1').innerHTML = highlightJson(steps.step_1_raw_extraction.raw_tokens);
            
            // 3. Populate Step 2 (Normalization)
            updateConfidenceBadge('conf-2', steps.step_2_normalization.normalization_confidence);
            document.getElementById('result-2').innerHTML = highlightJson(steps.step_2_normalization.normalized_amounts);

            // 4. Populate Step 3 (Classification)
            updateConfidenceBadge('conf-3', steps.step_3_classification.confidence);
            document.getElementById('result-3').innerHTML = highlightJson(steps.step_3_classification.amounts);

            // 5. Populate Step 4 (Final Structured Output)
            const currency = steps.step_4_final_result.currency || 'USD';
            document.getElementById('currency-4').textContent = currency;

            // Display two ways: structured HTML list and raw JSON output
            document.getElementById('final-summary-list').innerHTML = renderFinalAmountCards(steps.step_4_final_result.amounts, currency);
            document.getElementById('result-4-full').innerHTML = highlightJson(steps);


            // Ensure Step 4 is visible and others collapsed
            document.querySelectorAll('.step-content').forEach(el => el.classList.remove('active'));
            document.getElementById('step-4-full-content').classList.add('active'); // Keep full JSON open
            
        } else {
            // Handle API error/Guardrail output
            let reason = data.reason || (data.detail ? data.detail[0].msg : 'Unknown server error.');
            showError(`Processing Failed: ${reason}`);
        }

    } catch (error) {
        loadingOverlay.style.display = 'none';
        submitButton.disabled = false;
        showError(`❌ An unexpected network error occurred: ${error.message}`);
    }
}

function showError(message) {
    const errorMsg = document.getElementById('error-message');
    const initialMsg = document.getElementById('initial-message');
    
    errorMsg.textContent = message;
    errorMsg.classList.remove('hidden');
    initialMsg.classList.remove('hidden');
}


// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('extraction-form');
    form.addEventListener('submit', handleSubmit);
    
    // Set initial state
    document.getElementById('initial-message').classList.remove('hidden');
    
    // Add event listeners for step collapsing
    document.querySelectorAll('.step-header').forEach(header => {
        header.addEventListener('click', () => {
            const step = header.getAttribute('data-step');
            const content = document.getElementById(`step-${step}-content`);
            
            // Toggle visibility of the content
            content.classList.toggle('active');
        });
    });
});
