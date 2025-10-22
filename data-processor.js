class TaxiDataProcessor {
    constructor() {
        this.rawData = null;
        this.processedData = null;
        this.aggregatedData = null;
        this.sequences = null;
        this.trainData = null;
        this.testData = null;
        this.featureScaler = null;
        this.targetScaler = null;
        this.regions = new Map();
        
        console.log('✅ TaxiDataProcessor initialized');
    }

    async loadRawCSV(file) {
        return new Promise((resolve, reject) => {
            console.log('📁 Starting CSV parsing for file:', file.name);
            
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    console.log('✅ CSV parsing completed. Total rows:', results.data.length);
                    
                    if (results.errors.length > 0) {
                        console.warn('⚠️ CSV parsing warnings:', results.errors);
                    }
                    
                    if (!results.data || results.data.length === 0) {
                        reject(new Error('CSV file is empty or could not be parsed'));
                        return;
                    }
                    
                    // Filter out invalid records
                    this.rawData = results.data.filter(row => {
                        const isValid = row.pickup_datetime && 
                                       row.pickup_longitude && 
                                       row.pickup_latitude &&
                                       !isNaN(row.pickup_longitude) &&
                                       !isNaN(row.pickup_latitude);
                        return isValid;
                    });
                    
                    console.log(`✅ Valid records: ${this.rawData.length}/${results.data.length}`);
                    
                    if (this.rawData.length === 0) {
                        reject(new Error('No valid records found. Required columns: pickup_datetime, pickup_longitude, pickup_latitude'));
                        return;
                    }
                    
                    resolve(this.rawData);
                },
                error: (error) => {
                    console.error('❌ CSV parsing error:', error);
                    reject(new Error(`CSV parsing failed: ${error.message}`));
                }
            });
        });
    }

    processRawData(aggregationLevel = 'daily', regionSize = 0.05) {
        if (!this.rawData || this.rawData.length === 0) {
            throw new Error('No raw data loaded');
        }

        console.log(`🔄 Processing ${this.rawData.length} raw records...`);

        // Process each trip
        const processedTrips = this.rawData.map((trip, index) => {
            try {
                const pickupDate = new Date(trip.pickup_datetime);
                
                // Create region ID based on coordinates
                const regionId = this.getRegionId(
                    trip.pickup_longitude, 
                    trip.pickup_latitude, 
                    regionSize
                );

                return {
                    timestamp: pickupDate,
                    date: pickupDate.toISOString().split('T')[0],
                    hour: pickupDate.getHours(),
                    dayOfWeek: pickupDate.getDay(),
                    month: pickupDate.getMonth(),
                    region: regionId,
                    longitude: trip.pickup_longitude,
                    latitude: trip.pickup_latitude,
                    passenger_count: trip.passenger_count || 1
                };
            } catch (error) {
                console.warn(`Skipping invalid trip record ${index}:`, error);
                return null;
            }
        }).filter(trip => trip !== null);

        this.processedData = processedTrips;
        console.log(`✅ Processed ${this.processedData.length} trips`);
        return this.processedData;
    }

    getRegionId(longitude, latitude, regionSize) {
        const lonCell = Math.floor(longitude / regionSize);
        const latCell = Math.floor(latitude / regionSize);
        return `region_${lonCell}_${latCell}`;
    }

    aggregateData(aggregationLevel = 'daily') {
        if (!this.processedData) {
            throw new Error('No processed data available');
        }

        console.log('📊 Aggregating data...');

        // Group by date and region
        const groupedData = {};
        
        this.processedData.forEach(trip => {
            const key = aggregationLevel === 'daily' 
                ? `${trip.date}_${trip.region}`
                : `${trip.date}_${trip.hour.toString().padStart(2, '0')}_${trip.region}`;

            if (!groupedData[key]) {
                groupedData[key] = {
                    date: trip.date,
                    hour: trip.hour,
                    region: trip.region,
                    demand: 0,
                    totalPassengers: 0,
                    dayOfWeek: trip.dayOfWeek,
                    month: trip.month,
                    isWeekend: trip.dayOfWeek === 0 || trip.dayOfWeek === 6 ? 1 : 0
                };
            }

            groupedData[key].demand += 1;
            groupedData[key].totalPassengers += trip.passenger_count;
        });

        // Convert to array and sort by date
        this.aggregatedData = Object.values(groupedData)
            .sort((a, b) => new Date(a.date) - new Date(b.date));

        console.log(`✅ Aggregated ${this.aggregatedData.length} records`);
        return this.aggregatedData;
    }

    createSequences(sequenceLength = 14) {
        if (!this.aggregatedData || this.aggregatedData.length === 0) {
            throw new Error('No aggregated data available');
        }

        console.log('🔄 Creating sequences...');

        // Get unique regions
        const regions = [...new Set(this.aggregatedData.map(d => d.region))];
        const features = [];
        const targets = [];
        const dates = [];

        // For each region, create sequences
        regions.forEach(region => {
            const regionData = this.aggregatedData
                .filter(d => d.region === region)
                .sort((a, b) => new Date(a.date) - new Date(b.date));

            if (regionData.length <= sequenceLength) {
                console.log(`⚠️ Skipping region ${region}: insufficient data (${regionData.length} records)`);
                return;
            }

            for (let i = sequenceLength; i < regionData.length; i++) {
                const sequence = [];
                
                for (let j = i - sequenceLength; j < i; j++) {
                    const featureVector = [
                        regionData[j].demand,
                        regionData[j].dayOfWeek / 6, // Normalize day of week
                        regionData[j].isWeekend,
                        regionData[j].totalPassengers / Math.max(1, regionData[j].demand) // avg passengers per trip
                    ];
                    sequence.push(featureVector);
                }
                
                features.push(sequence);
                targets.push(regionData[i].demand);
                dates.push({
                    date: regionData[i].date,
                    region: regionData[i].region
                });
            }
        });

        if (features.length === 0) {
            throw new Error('No sequences could be created. Check sequence length and data size.');
        }

        this.sequences = {
            features: tf.tensor3d(features),
            targets: tf.tensor1d(targets),
            dates: dates,
            regions: regions
        };

        console.log(`✅ Created ${features.length} sequences with shape: [${features.length}, ${sequenceLength}, 4]`);
        return this.sequences;
    }

    splitData(trainRatio = 0.8) {
        if (!this.sequences) {
            throw new Error('No sequences created');
        }

        const totalSamples = this.sequences.features.shape[0];
        const trainSize = Math.floor(totalSamples * trainRatio);
        
        console.log(`📊 Splitting data: ${trainSize} training, ${totalSamples - trainSize} test samples`);

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

        return { 
            train: this.trainData, 
            test: this.testData,
            featureCount: this.sequences.features.shape[2],
            sequenceLength: this.sequences.features.shape[1]
        };
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

    generateFutureSequence(daysToPredict = 7) {
        if (!this.aggregatedData || !this.sequences) {
            throw new Error('No data available for future prediction');
        }

        // Get the most recent sequence for each region
        const regions = this.sequences.regions;
        const sequenceLength = this.sequences.features.shape[1];
        const futureSequences = [];
        const futureDates = [];

        const lastDate = new Date(this.aggregatedData[this.aggregatedData.length - 1].date);

        regions.forEach(region => {
            const regionData = this.aggregatedData
                .filter(d => d.region === region)
                .sort((a, b) => new Date(a.date) - new Date(b.date));

            if (regionData.length < sequenceLength) return;

            // Get the most recent sequence
            const lastSequence = regionData.slice(-sequenceLength);
            
            // Create base sequence for prediction
            const baseSequence = lastSequence.map(day => [
                day.demand,
                day.dayOfWeek / 6,
                day.isWeekend,
                day.totalPassengers / Math.max(1, day.demand)
            ]);

            futureSequences.push(baseSequence);

            // Generate future dates
            const regionFutureDates = [];
            for (let i = 1; i <= daysToPredict; i++) {
                const futureDate = new Date(lastDate);
                futureDate.setDate(lastDate.getDate() + i);
                regionFutureDates.push({
                    date: futureDate.toISOString().split('T')[0],
                    region: region
                });
            }
            futureDates.push(regionFutureDates);
        });

        if (futureSequences.length === 0) {
            throw new Error('Could not generate future sequences');
        }

        return {
            features: this.featureScaler.normalize(tf.tensor3d(futureSequences)),
            dates: futureDates.flat()
        };
    }

    getDataSummary() {
        const summary = {
            rawRecords: this.rawData ? this.rawData.length : 0,
            processedTrips: this.processedData ? this.processedData.length : 0,
            aggregatedRecords: this.aggregatedData ? this.aggregatedData.length : 0,
            sequences: this.sequences ? this.sequences.features.shape[0] : 0,
            regions: this.sequences ? this.sequences.regions.length : 0
        };

        if (this.aggregatedData && this.aggregatedData.length > 0) {
            summary.dateRange = {
                start: this.aggregatedData[0].date,
                end: this.aggregatedData[this.aggregatedData.length - 1].date
            };
            summary.avgDemand = this.aggregatedData.reduce((sum, row) => sum + row.demand, 0) / this.aggregatedData.length;
        }

        return summary;
    }

    getDataPreview(limit = 10) {
        if (!this.aggregatedData) return [];
        
        return this.aggregatedData.slice(0, Math.min(limit, this.aggregatedData.length)).map(row => ({
            date: row.date,
            region: row.region,
            demand: row.demand,
            dayOfWeek: row.dayOfWeek,
            isWeekend: row.isWeekend ? 'Yes' : 'No'
        }));
    }

    dispose() {
        console.log('🧹 Cleaning up TaxiDataProcessor...');
        
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
        if (this.featureScaler) {
            this.featureScaler.min.dispose();
            this.featureScaler.max.dispose();
        }
        if (this.targetScaler) {
            this.targetScaler.min.dispose();
            this.targetScaler.max.dispose();
        }
    }
}