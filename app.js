// Global instances
let dataProcessor = null;
let demandModel = null;

// Initialize application
function initApp() {
    try {
        console.log('🚕 Initializing Taxi Demand Forecasting App...');
        
        // Initialize classes
        dataProcessor = new TaxiDataProcessor();
        demandModel = new TaxiDemandModel();
        
        // Set up event listeners
        setupEventListeners();
        
        updateUIState();
        updateDataStatus('✅ Application ready! Select a CSV file to begin.');
        
        console.log('✅ App initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize app:', error);
        updateDataStatus('❌ Failed to initialize application', true);
    }
}

function setupEventListeners() {
    // File input change listener
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                console.log('📁 File selected:', file.name);
                updateDataStatus(`📁 Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
            }
        });
    }
}

// UI Update functions
function updateDataStatus(message, isError = false) {
    const statusElement = document.getElementById('dataStatus');
    if (statusElement) {
        statusElement.innerHTML = message;
        statusElement.className = isError ? 'status-error' : 'status-success';
    }
}

function updateTrainingProgress(progress) {
    const progressElement = document.getElementById('trainingProgress');
    if (progressElement && progress) {
        const percentage = Math.round((progress.epoch / progress.totalEpochs) * 100);
        
        let progressHTML = `<div><strong>Epoch ${progress.epoch}/${progress.totalEpochs} (${percentage}%)</strong></div>`;
        progressHTML += `<div>Loss: ${progress.loss.toFixed(4)} | MAE: ${progress.mae.toFixed(4)}</div>`;
        
        if (progress.valLoss) {
            progressHTML += `<div>Val Loss: ${progress.valLoss.toFixed(4)} | Val MAE: ${progress.valMae.toFixed(4)}</div>`;
        }
        
        progressElement.innerHTML = progressHTML;
    }
}

function updateUIState() {
    const trainBtn = document.getElementById('trainBtn');
    const evalBtn = document.getElementById('evalBtn');
    const predictBtn = document.getElementById('predictBtn');
    const saveBtn = document.getElementById('saveBtn');
    
    const hasTrainData = dataProcessor && dataProcessor.trainData;
    const isModelTrained = demandModel && demandModel.isTrained;
    
    if (trainBtn) trainBtn.disabled = !hasTrainData;
    if (evalBtn) evalBtn.disabled = !isModelTrained;
    if (predictBtn) predictBtn.disabled = !isModelTrained;
    if (saveBtn) saveBtn.disabled = !isModelTrained;
}

// Data processing functions
async function loadRawData() {
    if (!dataProcessor) {
        updateDataStatus('❌ Application not initialized', true);
        return;
    }

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        updateDataStatus('❌ Please select a CSV file', true);
        return;
    }

    try {
        updateDataStatus('📁 Loading CSV data...');
        console.log('Loading file:', file.name);
        
        await dataProcessor.loadRawCSV(file);
        
        const summary = dataProcessor.getDataSummary();
        console.log('Data summary:', summary);
        
        updateDataStatus(`
            ✅ Data loaded successfully!<br>
            📊 Records: ${summary.rawRecords.toLocaleString()}<br>
            📄 File: ${file.name}
        `);
        
    } catch (error) {
        console.error('Error loading data:', error);
        updateDataStatus(`❌ Error: ${error.message}`, true);
    }
}

async function processRawData() {
    if (!dataProcessor) {
        updateDataStatus('❌ Application not initialized', true);
        return;
    }

    try {
        const aggregationLevel = document.getElementById('aggregationLevel').value;
        const regionSize = parseFloat(document.getElementById('regionSize').value);
        
        updateDataStatus('🔄 Processing data...');
        
        dataProcessor.processRawData(aggregationLevel, regionSize);
        dataProcessor.aggregateData(aggregationLevel);
        
        const summary = dataProcessor.getDataSummary();
        const preview = dataProcessor.getDataPreview(5);
        
        // Display preview
        let previewHTML = '<h4>Data Preview:</h4><table><tr><th>Date</th><th>Region</th><th>Demand</th><th>Day</th><th>Weekend</th></tr>';
        
        if (preview && preview.length > 0) {
            preview.forEach(row => {
                previewHTML += `<tr>
                    <td>${row.date}</td>
                    <td>${row.region}</td>
                    <td>${row.demand}</td>
                    <td>${row.dayOfWeek}</td>
                    <td>${row.isWeekend}</td>
                </tr>`;
            });
        } else {
            previewHTML += '<tr><td colspan="5">No data available</td></tr>';
        }
        
        previewHTML += '</table>';
        document.getElementById('dataPreview').innerHTML = previewHTML;
        
        updateDataStatus(`
            ✅ Data processed!<br>
            🚕 Trips: ${summary.processedTrips ? summary.processedTrips.toLocaleString() : 0}<br>
            📈 Aggregated: ${summary.aggregatedRecords ? summary.aggregatedRecords.toLocaleString() : 0}<br>
            ${summary.dateRange ? `📅 From ${summary.dateRange.start} to ${summary.dateRange.end}` : ''}
        `);
        
    } catch (error) {
        console.error('Error processing data:', error);
        updateDataStatus(`❌ Error: ${error.message}`, true);
    }
}

async function prepareTrainingData() {
    if (!dataProcessor) {
        updateDataStatus('❌ Application not initialized', true);
        return;
    }

    try {
        const sequenceLength = parseInt(document.getElementById('sequenceLength').value);
        const trainRatio = parseFloat(document.getElementById('trainRatio').value);
        
        updateDataStatus('🔄 Creating sequences...');
        
        dataProcessor.createSequences(sequenceLength);
        const splitResult = dataProcessor.splitData(trainRatio);
        
        const summary = dataProcessor.getDataSummary();
        
        updateDataStatus(`
            ✅ Training data ready!<br>
            🔢 Sequences: ${summary.sequences}<br>
            🗺️ Regions: ${summary.regions}<br>
            📐 Input shape: [${splitResult.sequenceLength}, ${splitResult.featureCount}]
        `);
        
        updateUIState();
        
    } catch (error) {
        console.error('Error preparing data:', error);
        updateDataStatus(`❌ Error: ${error.message}`, true);
    }
}

// Model functions
async function createModel() {
    if (!dataProcessor || !demandModel) {
        updateDataStatus('❌ Application not initialized', true);
        return;
    }

    try {
        if (!dataProcessor.sequences) {
            throw new Error('Please prepare training data first');
        }
        
        const sequenceLength = dataProcessor.sequences.features.shape[1];
        const featureCount = dataProcessor.sequences.features.shape[2];
        const units = parseInt(document.getElementById('rnnUnits').value);
        
        demandModel.createModel(sequenceLength, featureCount, units);
        updateDataStatus('✅ RNN model created successfully!');
        updateUIState();
        
    } catch (error) {
        console.error('Error creating model:', error);
        updateDataStatus(`❌ Error: ${error.message}`, true);
    }
}

async function trainModel() {
    if (!dataProcessor || !demandModel) {
        updateDataStatus('❌ Application not initialized', true);
        return;
    }

    try {
        if (!demandModel.model || !dataProcessor.trainData) {
            throw new Error('Please create model and prepare data first');
        }
        
        const epochs = parseInt(document.getElementById('epochs').value);
        
        updateDataStatus('🎯 Training started...');
        
        await demandModel.train(
            dataProcessor.trainData.features, 
            dataProcessor.trainData.targets,
            epochs
        );
        
        updateDataStatus('✅ Training completed!');
        updateUIState();
        
    } catch (error) {
        console.error('Training error:', error);
        updateDataStatus(`❌ Training error: ${error.message}`, true);
    }
}

async function evaluateModel() {
    if (!dataProcessor || !demandModel) {
        updateDataStatus('❌ Application not initialized', true);
        return;
    }

    try {
        if (!demandModel.isTrained || !dataProcessor.testData) {
            throw new Error('Model not trained or test data not available');
        }
        
        updateDataStatus('📊 Evaluating model...');
        
        // Get predictions
        const predictions = await demandModel.predict(dataProcessor.testData.features);
        const denormalizedPreds = dataProcessor.targetScaler.denormalize(predictions);
        
        // Calculate metrics
        const evaluation = await demandModel.evaluate(dataProcessor.testData.features, dataProcessor.testData.targets);
        const maeDenormalized = await calculateMAE(denormalizedPreds, dataProcessor.testData.originalTargets);
        
        // Calculate accuracy (1 - normalized MAE)
        const accuracy = Math.max(0, (1 - evaluation.mae) * 100);
        
        // Display results
        const metricsElement = document.getElementById('performanceMetrics');
        if (metricsElement) {
            metricsElement.innerHTML = `
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff;">
                    <h3 style="margin-top: 0; color: #007bff;">📊 Model Performance Results</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                        <div style="background: white; padding: 15px; border-radius: 6px; border: 1px solid #e9ecef;">
                            <strong>🎯 Accuracy</strong><br>
                            <span style="font-size: 24px; color: #28a745;">${accuracy.toFixed(1)}%</span>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 6px; border: 1px solid #e9ecef;">
                            <strong>📈 Normalized MAE</strong><br>
                            <span style="font-size: 18px;">${evaluation.mae.toFixed(4)}</span>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 6px; border: 1px solid #e9ecef;">
                            <strong>🚕 Actual MAE</strong><br>
                            <span style="font-size: 18px;">${maeDenormalized.toFixed(2)} rides</span>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 6px; border: 1px solid #e9ecef;">
                            <strong>🔢 Test Samples</strong><br>
                            <span style="font-size: 18px;">${dataProcessor.testData.originalTargets.shape[0]}</span>
                        </div>
                    </div>
                    ${dataProcessor.testData.dates && dataProcessor.testData.dates.length > 0 ? 
                        `<div style="margin-top: 15px; padding: 10px; background: #e7f3ff; border-radius: 4px;">
                            <strong>📅 Date Range:</strong> ${dataProcessor.testData.dates[0].date} to ${dataProcessor.testData.dates[dataProcessor.testData.dates.length - 1].date}
                        </div>` : ''}
                </div>
            `;
        }
        
        // Clean up
        predictions.dispose();
        denormalizedPreds.dispose();
        
        updateDataStatus('✅ Evaluation completed!');
        
    } catch (error) {
        console.error('Evaluation error:', error);
        updateDataStatus(`❌ Evaluation error: ${error.message}`, true);
    }
}

async function predictFuture() {
    if (!dataProcessor || !demandModel) {
        updateDataStatus('❌ Application not initialized', true);
        return;
    }

    try {
        if (!demandModel.isTrained) {
            throw new Error('Model not trained');
        }
        
        updateDataStatus('🔮 Generating future predictions...');
        
        const futureData = dataProcessor.generateFutureSequence(7);
        const futurePredictions = await demandModel.predict(futureData.features);
        const denormalizedFuture = dataProcessor.targetScaler.denormalize(futurePredictions);
        
        const futureValues = await denormalizedFuture.data();
        
        // Display future predictions
        const futureElement = document.getElementById('futurePredictions');
        if (futureElement) {
            let futureHTML = `
                <div style="background: #fff3cd; padding: 20px; border-radius: 8px; border-left: 4px solid #ffc107; margin-top: 20px;">
                    <h3 style="margin-top: 0; color: #856404;">🔮 Next 7 Days Predictions</h3>
                    <table style="width: 100%; background: white; border-radius: 6px; overflow: hidden;">
                        <thead>
                            <tr style="background: #f8f9fa;">
                                <th style="padding: 12px; border-bottom: 2px solid #dee2e6;">Date</th>
                                <th style="padding: 12px; border-bottom: 2px solid #dee2e6;">Region</th>
                                <th style="padding: 12px; border-bottom: 2px solid #dee2e6;">Predicted Demand</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            if (futureData.dates && futureData.dates.length > 0) {
                futureData.dates.forEach((dateObj, index) => {
                    if (index < futureValues.length) {
                        futureHTML += `
                            <tr style="border-bottom: 1px solid #dee2e6;">
                                <td style="padding: 12px;">${dateObj.date}</td>
                                <td style="padding: 12px;">${dateObj.region}</td>
                                <td style="padding: 12px; font-weight: bold; color: #28a745;">${Math.round(futureValues[index])} rides</td>
                            </tr>
                        `;
                    }
                });
            }
            
            futureHTML += `
                        </tbody>
                    </table>
                </div>
            `;
            futureElement.innerHTML = futureHTML;
        }
        
        // Clean up
        futurePredictions.dispose();
        denormalizedFuture.dispose();
        
        updateDataStatus('✅ Future predictions generated!');
        
    } catch (error) {
        console.error('Prediction error:', error);
        updateDataStatus(`❌ Prediction error: ${error.message}`, true);
    }
}

// Utility functions
async function calculateMAE(predictions, actual) {
    if (!predictions || !actual) return 0;
    
    try {
        const absoluteErrors = tf.abs(tf.sub(predictions, actual));
        const mae = await absoluteErrors.mean().data();
        absoluteErrors.dispose();
        return mae[0];
    } catch (error) {
        console.error('Error calculating MAE:', error);
        return 0;
    }
}

async function saveModel() {
    if (!demandModel) {
        updateDataStatus('❌ Application not initialized', true);
        return;
    }

    try {
        await demandModel.saveModel();
        updateDataStatus('✅ Model saved successfully! You can download the model files.');
    } catch (error) {
        console.error('Error saving model:', error);
        updateDataStatus(`❌ Error saving model: ${error.message}`, true);
    }
}

async function loadModel() {
    updateDataStatus('ℹ️ Model loading feature to be implemented in next version');
}

// Memory management
function cleanup() {
    if (dataProcessor) dataProcessor.dispose();
    if (demandModel) demandModel.dispose();
    tf.disposeVariables();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initApp);

// Cleanup
window.addEventListener('beforeunload', cleanup);