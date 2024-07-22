// JavaScript code to handle classification and rendering of predictions and chart

// Function to handle classification logic when the "Classify" button is clicked
async function classifyDescriptions() {
  // Collect the form data from the textarea
  const descriptions = document.getElementById('descriptions').value.split('\n').filter(Boolean);
  
  // Create the JSON payload from descriptions entered by the user
  const data = descriptions.map(description => {
      const fields = description.split(',').map(field => field.trim()); // Trim whitespace from each field
      return {
          "Customer ID": parseInt(fields[0]),                      // Parse Customer ID as integer
          "Age": parseInt(fields[1]),                              // Parse Age as integer
          "Gender": fields[2],                                     // Gender field as is
          "Item Purchased": fields[3],                             // Item Purchased field as is
          "Purchase Amount (USD)": parseFloat(fields[5]),          // Parse Purchase Amount (USD) as float
          "Location": fields[6],                                   // Location field as is
          "Size": fields[7],                                       // Size field as is
          "Color": fields[8],                                      // Color field as is
          "Season": fields[9],                                     // Season field as is
          "Review Rating": parseFloat(fields[10]),                 // Parse Review Rating as float
          "Subscription Status": fields[11],                       // Subscription Status field as is
          "Shipping Type": fields[12],                              // Shipping Type field as is
          "Discount Applied": fields[13],                          // Discount Applied field as is
          "Promo Code Used": fields[14],                            // Promo Code Used field as is
          "Previous Purchases": parseInt(fields[15]),              // Parse Previous Purchases as integer
          "Payment Method": fields[16],                            // Payment Method field as is
          "Frequency of Purchases": fields[17]                     // Frequency of Purchases field as is
      };
  });

  console.log('Sending data:', data);  // Log the data being sent for debugging

  // Send the JSON payload to the Flask backend for classification
  const response = await fetch('http://127.0.0.1:5001/classify', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
  });

  // Handle the response from the backend
  const result = await response.json();
  console.log('Received response:', result);  // Log the response received for debugging

  // If predictions are received, display them and create a bar chart
  if (result.predictions) {
      displayPredictions(result.predictions);   // Function to display predictions in HTML
      createBarChart(result.predictions);       // Function to create a bar chart using D3.js
  } else {
      console.error('No predictions found in the response:', result);  // Log an error if no predictions are found
  }
}

// Function to display predictions in HTML
function displayPredictions(predictions) {
  const predictionsDiv = document.getElementById('predictions');
  predictionsDiv.innerHTML = '<pre>' + JSON.stringify(predictions, null, 2) + '</pre>';  // Display predictions in a <pre> element
}

// Function to create a bar chart using D3.js
function createBarChart(predictions) {
  // Count occurrences of each prediction
  const data = {};
  predictions.forEach(pred => {
      data[pred] = (data[pred] || 0) + 1;
  });

  // Clear previous chart if exists
  const chartDiv = document.getElementById('chart');
  chartDiv.innerHTML = ''; 

  // Define chart dimensions and margins
  const margin = { top: 20, right: 30, bottom: 40, left: 40 },
      width = 600 - margin.left - margin.right,
      height = 400 - margin.top - margin.bottom;

  // Create SVG element for the chart
  const svg = d3.select("#chart")
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

  // Define scales for x and y axes
  const x = d3.scaleBand()
      .domain(Object.keys(data))
      .range([0, width])
      .padding(0.1);

  const y = d3.scaleLinear()
      .domain([0, d3.max(Object.values(data))])
      .nice()
      .range([height, 0]);

  // Add bars to the chart
  svg.append("g")
      .selectAll(".bar")
      .data(Object.entries(data))
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", d => x(d[0]))
      .attr("y", d => y(d[1]))
      .attr("width", x.bandwidth())
      .attr("height", d => height - y(d[1]));

  // Add x-axis to the chart
  svg.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x));

  // Add y-axis to the chart
  svg.append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(y));
}
