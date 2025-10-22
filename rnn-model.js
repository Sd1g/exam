class TaxiDemandModel {
    constructor() {
        this.model = null;
        this.history = {
            epochs: [],
            history: {
                loss: [],
                val_loss: [],
                mae: [],
                val_mae: []
            }
        };
        this.isTrained = false;
        this.modelConfig = {};
    }

    createModel(sequenceLength, featureCount, units = 64) {
        this.modelConfig = { sequenceLength, featureCount, units };
        
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: units,
                    returnSequences: true,
                    inputShape: [sequenceLength, featureCount],
                    recurrentInitializer: 'glorotNormal'
                }),
                tf.layers.dropout({ rate: 0.3 }),
                
                tf.layers.gru({
                    units: Math.floor(units / 2),
                    returnSequences: false,
                    recurrentInitializer: 'glorotNormal'
                }),
                tf.layers.dropout({ rate: 0.2 }),
                
                tf.layers.dense({ 
                    units: Math.floor(units / 4), 
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                
                tf.layers.dense({ 
                    units: 1, 
                    activation: 'linear',
                    kernelInitializer: 'heNormal'
                })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        console.log('✅ Model created successfully');
        
        return this.model;
    }

    async train(X_train, y_train, epochs = 50, validationSplit = 0.2) {
        if (!this.model) throw new Error('Model not created');
        
        console.log(`🎯 Training model for ${epochs} epochs...`);
        
        // Reset history
        this.history = {
            epochs: [],
            history: {
                loss: [],
                val_loss: [],
                mae: [],
                val_mae: []
            }
        };
        
        const history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: 32,
            validationSplit: validationSplit,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    // Store history for progress updates
                    this.history.epochs.push(epoch);
                    this.history.history.loss.push(logs.loss);
                    this.history.history.mae.push(logs.mae);
                    this.history.history.val_loss.push(logs.val_loss);
                    this.history.history.val_mae.push(logs.val_mae);
                    
                    const progress = {
                        epoch: epoch + 1,
                        totalEpochs: epochs,
                        loss: logs.loss,
                        mae: logs.mae,
                        valLoss: logs.val_loss,
                        valMae: logs.val_mae
                    };
                    
                    console.log(`Epoch ${epoch + 1}/${epochs}: loss=${logs.loss.toFixed(4)}, mae=${logs.mae.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}, val_mae=${logs.val_mae.toFixed(4)}`);
                    
                    if (typeof updateTrainingProgress === 'function') {
                        updateTrainingProgress(progress);
                    }
                },
                onTrainEnd: () => {
                    console.log('✅ Training completed');
                }
            }
        });

        this.isTrained = true;
        return this.history;
    }

    async predict(features) {
        if (!this.model || !this.isTrained) {
            throw new Error('Model not trained');
        }
        
        console.log('🔮 Making predictions...');
        return this.model.predict(features);
    }

    async evaluate(X_test, y_test) {
        if (!this.model || !this.isTrained) {
            throw new Error('Model not trained');
        }
        
        console.log('📊 Evaluating model...');
        const evaluation = this.model.evaluate(X_test, y_test);
        const loss = evaluation[0].dataSync()[0];
        const mae = evaluation[1].dataSync()[0];
        
        // Clean up
        evaluation.forEach(tensor => tensor.dispose());
        
        return { loss, mae };
    }

    async saveModel() {
        if (!this.model) throw new Error('No model to save');
        
        console.log('💾 Saving model...');
        const saveResult = await this.model.save('downloads://taxi-demand-model');
        return saveResult;
    }

    getModelInfo() {
        if (!this.model) return 'No model created';
        
        return {
            layers: this.model.layers.length,
            trained: this.isTrained,
            config: this.modelConfig
        };
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}