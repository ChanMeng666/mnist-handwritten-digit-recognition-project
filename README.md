# MNIST Neural Network Analysis Project

## Overview
This project presents a comprehensive analysis of the MNIST handwritten digit recognition problem using three different neural network approaches. The implementation includes basic MLP, feature-optimized, and deep CNN models, along with extensive performance analysis and model comparisons.

## Project Highlights
- Comprehensive data exploration and visualization
- Implementation of three distinct neural network architectures
- Extensive feature importance analysis
- Detailed sensitivity and robustness testing
- Performance comparison and trade-off analysis
- thorough error analysis and visualization

## Models Implemented
1. **Basic Model**
   - Simple MLP architecture
   - Tested with various neuron configurations (128, 256, 512)
   - Achieved 99.05% accuracy

2. **Optimized Model**
   - Feature-selected architecture
   - Reduced input dimensionality (196 features)
   - Resource-efficient implementation
   - Achieved 97.86% accuracy

3. **Deep Model**
   - Convolutional Neural Network
   - Multiple convolutional layers with batch normalization
   - Advanced learning rate scheduling
   - Achieved 99.71% accuracy

## Key Features
- Data augmentation pipeline
- Feature importance visualization
- Occlusion sensitivity analysis
- Performance metrics comparison
- Comprehensive error analysis
- Model robustness evaluation

## Results Summary
| Model     | Accuracy | Prediction Time | Parameters |
| --------- | -------- | --------------- | ---------- |
| Basic     | 99.05%   | 0.621s          | 407,050    |
| Optimized | 97.86%   | 0.528s          | 84,618     |
| Deep      | 99.71%   | 4.869s          | 1,015,530  |

## Technologies Used
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Key Findings
- Deep CNN model achieves the highest accuracy but requires more computational resources
- Optimized model offers the best balance of performance and resource utilization
- Feature selection significantly reduces model complexity while maintaining acceptable accuracy
- Central image regions show highest importance for digit recognition

## Future Improvements
- Implementation of additional neural network architectures
- Enhanced data augmentation techniques
- Model compression and optimization
- Real-time prediction capabilities
- Transfer learning experiments

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author
[Chan Meng]: https://github.com/ChanMeng666

## Acknowledgments
- The MNIST database of handwritten digits
- TensorFlow and Keras documentation
- Scientific Python community
