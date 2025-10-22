class TaxiDataLoader {
    constructor() {
        this.rawData = null;
        this.processedData = null;
        this.sequences = null;
        this.trainData = null;
        this.testData = null;
        this.featureScaler = null;
        this.targetScaler = null;
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(new Error(`CSV parsing errors: ${results.errors[0].message}`));
                        return;
                    }
                    this.rawData = results.data;
                    resolve(this.rawData);
                },
                error: (error) => reject(error)
            });
        });
    }

    preprocessData() {
        if (!this.rawData) throw new Error('No data loaded');
        
        // Filter out invalid records
        const validData = this.rawData.filter(row => 
            row.date && row.demand && !isNaN(row.demand) && row.demand > 0
        );

        // Sort by date
        validData.sort((a, b) => new Date(a.date) - new Date(b.date));

        // Add time-based features
        this.processedData = validData.map(row => {
            const date = new Date(row.date);
            return {
                date: row.date,
                demand: row.demand,
                dayOfWeek: date.getDay(),
                dayOfMonth: date.getDate(),
                month: date.getMonth(),
                isWeekend: date.getDay() === 0 || date.getDay() === 6 ? 1 : 0,
                // Add more features like holidays, weather if available
                ...row
            };
        });

        return this.processedData;
    }

    createSequences(sequenceLength = 14) {
        if (!this.processedData) throw new Error('No processed data available');
        
        const features = [];
        const targets = [];
        const dates = [];

        // Use multiple features: demand, dayOfWeek, isWeekend
        const featureColumns = ['demand', 'dayOfWeek', 'isWeekend'];
        
        for (let i = sequenceLength; i < this.processedData.length; i++) {
            const sequence = [];
            
            for (let j = i - sequenceLength; j < i; j++) {
                const featureVector = featureColumns.map(col => this.processedData[j][col]);
                sequence.push(featureVector);
            }
            
            features.push(sequence);
            targets.push(this.processedData[i].demand);
            dates.push(this.processedData[i].date);
        }

        this.sequences = {
            features: tf.tensor3d(features),
            targets: tf.tensor1d(targets),
            dates: dates
        };

        return this.sequences;
    }

    splitData(trainRatio = 0.8) {
        if (!this.sequences) throw new Error('No sequences created');
        
        const totalSamples = this.sequences.features.shape[0];
        const trainSize = Math.floor(totalSamples * trainRatio);
        
        const X_train = this.sequences.features.slice([0, 0, 0], [trainSize, -1, -1]);
        const y_train = this.sequences.targets.slice([0], [trainSize]);
        const X_test = this.sequences.features.slice([trainSize, 0, 0], [-1, -1, -1]);
        const y_test = this.sequences.targets.slice([trainSize], [-1]);
        const test_dates = this.sequences.dates.slice(trainSize);

        // Normalize features
        this.featureScaler = this.createScaler(X_train);
        this.targetScaler = this.createScaler(y_train);

        this.trainData = {
            features: this.featureScaler.normalize(X_train),
            targets: this.targetScaler.normalize(y_train)
        };

        this.testData = {
            features: this.featureScaler.normalize(X_test),
            targets: this.targetScaler.normalize(y_test),
            dates: test_dates,
            originalTargets: y_test
        };

        return { train: this.trainData, test: this.testData };
    }

    createScaler(tensor) {
        const min = tensor.min();
        const max = tensor.max();
        
        return {
            min: min,
            max: max,
            normalize: (x) => x.sub(min).div(max.sub(min)),
            denormalize: (x) => x.mul(max.sub(min)).add(min)
        };
    }

    getDataSummary() {
        if (!this.processedData) return 'No data available';
        
        return {
            totalRecords: this.processedData.length,
            dateRange: {
                start: this.processedData[0].date,
                end: this.processedData[this.processedData.length - 1].date
            },
            avgDemand: this.processedData.reduce((sum, row) => sum + row.demand, 0) / this.processedData.length,
            sequences: this.sequences ? this.sequences.features.shape[0] : 0
        };
    }

    dispose() {
        if (this.sequences) {
            this.sequences.features.dispose();
            this.sequences.targets.dispose();
        }
        if (this.trainData) {
            this.trainData.features.dispose();
            this.trainData.targets.dispose();
        }
        if (this.testData) {
            this.testData.features.dispose();
            this.testData.targets.dispose();
            this.testData.originalTargets.dispose();
        }
    }
}