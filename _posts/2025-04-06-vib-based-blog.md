---
layout: post
title: Remaining Useful Life Estimation for IM using ML Approach
---
## Week 6 Update

Rotating machines has become a crucial component for many industrial processes today and failure of this component has a great impact on the overall performance of industries.
Upon attaining the vibrations signal of the bearing in the x and y directions, the data was explored using the MATLAB software and the code given below. The vibrations signal were attained as mlx files and has
to be loaded first into MATLAB's workspace for further exploration of the data. The two signals are plotted against time as well for further visualization labelled as timestamps. The timestamps was also analyzed 
in order to convert it to datetime format for easier labelling of data collection allocation for which particular day and hour. 

```matlab
load x_vib_signal_bearing.mat;
load y_vib_signal_bearing.mat;
load timestamps.mat;

plot(A)
plot(B)

% Convert the entire column to datetime
date = datetime(all_timestamps, 'InputFormat', 'yyyy-MM-dd_HH-mm');
u = unique(date);

% Step 1: Extract unique dates from the datetime array
uniqueDates = unique(dateshift(u, 'start', 'day'));

% Step 2: Initialize a count array to hold the counts for each unique date
counts = zeros(size(uniqueDates));

% Step 3: Count occurrences of each unique date
for i = 1:length(uniqueDates)
counts(i) = sum(dateshift(u, 'start', 'day') == uniqueDates(i));
end

% Step 4: Create a table with the unique dates and their counts
dateCountsTable = table(uniqueDates', counts', 'VariableNames', {'Date', 'Count'});

% Step 5: Create a new column where Count is multiplied by 796
dateCountsTable.NewCount = dateCountsTable.Count * 796;
```
<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/x-vib-signal.png" alt="X-Vibration Signal" title="X-Vibration Signal" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

Depcited above is the plot for the x acceleration vibration signal for all the bearing conditions from healthy state to the faulty state.
It can be seen that the amplitude of the signal increases after a few points in time indicating the change in condition of the bearing.

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/x-vib-loads.png" alt="X-Vibration Signal" title="X-Vibration Signal with Different Loading Conditions" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

Shown above is the x vibration signal with different colour coding depicting the different days at which the data was collected as well as separating the different load conditions at which the bearing was operated under. 

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/y-vib-signal.png" alt="Y-Vibration Signal" title="Y-Vibration Signal" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

Illustrated above is the y acceleration vibration signal for all the conditions of the bearing from a healthy state to the faulty state.
The y vibration signal also shows an increase in amplitude until failure. 

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/y-vib-loads.png" alt="Y-Vibration Signal" title="Y-Vibration Signal with Different Loading Conditions" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

Depicted above is also the y vibration signal with the colour coding, differentiating the different days at which the vibration data has been collected as well as the different loads at which the rotating machine was operated under. 

## Week 7 Update

Moreover, we used a function file shown below to extract 15 features from both the x and y vibrations signals by calling the function in the MATLAB main file:

```matlab

function statistical_features = stats(data, window_size, overlap)
    % Initialize an empty array to hold the statistical features
    statistical_features = [];

    % Initialize a figure for plotting
    figure;
    num_features = 15; % Total number of features
    num_rows = 5;      % Number of rows in the subplot
    num_cols = 3;      % Number of columns in the subplot

    % Loop through the data with the specified overlap
    for i = 1:overlap:(size(data, 1) - window_size + 1)
        window = data(i:i + window_size - 1, :);
        
        % Calculate statistical features
        Feature_mean = mean(window);
        Feature_max = max(window);
        Feature_rms = rms(window);
        Feature_srm = (sum(sqrt(abs(window))) / window_size)^2;
        Feature_std = std(window);
        Feature_var = var(window);
        Feature_rmsshape = Feature_rms / ((sum(abs(window))) / window_size);
        Feature_srmshape = Feature_srm / ((sum(abs(window))) / window_size);
        Feature_crestfact = Feature_max / Feature_rms;
        Feature_latfact = Feature_max / Feature_srm;
        Feature_impulsefact = Feature_max / ((sum(abs(window))) / window_size);
        Feature_skewness = skewness(window);
        Feature_kurtosis = kurtosis(window);
        Feature_5thM = moment(window, 5) / (std(window)^5);
        Feature_6thM = moment(window, 6) / (std(window)^6);

        % Combine features into a single row
        Combined = [Feature_mean, Feature_max, Feature_rms, Feature_srm, Feature_std, Feature_var, ...
                    Feature_rmsshape, Feature_srmshape, Feature_crestfact, Feature_latfact, ...
                    Feature_impulsefact, Feature_skewness, Feature_kurtosis, Feature_5thM, Feature_6thM];

        % Append to statistical features
        statistical_features = [statistical_features; Combined];
    end

    % Create a table with column names
    feature_names = {'Mean', 'Max', 'RMS', 'SRM', 'Std', 'Var', ...
                     'RMSShape', 'SRMShape', 'CrestFactor', 'LateralFactor', ...
                     'ImpulseFactor', 'Skewness', 'Kurtosis', '5thMoment', '6thMoment'};
    
    statistical_features = array2table(statistical_features, 'VariableNames', feature_names);

    % Plot each feature in a 5x3 grid
    for j = 1:num_features
        subplot(num_rows, num_cols, j);
        plot(statistical_features{:, j});
        title(feature_names{j});
        xlabel('Window Index');
        ylabel(feature_names{j});
        grid on;
    end

    % Adjust layout
    sgtitle('Statistical Features'); % Add a super title for the entire figure
end
```
15 statistical time-domain features were extracted from the raw data of x and y vibration signals individually at first. 

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/x-stats.png" alt="X Statistical Features" title="X Statistical Time-Domain Features" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

Illustrated above are the 15 time-domain statistical features for the x vibration signal. This includes the mean, max, rms, srm, std, var, rms shape, srm shape, crest factor, lareral factor, impulse factor, skewness, kurtosis, 5th moment, and the 6th moment. 

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/y-stats.png" alt="Y Statistical Features" title="Y Statistical Time-Domain Features" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

The figure above shows the 15 statistical time-domain features for the y vibration signal. 

This was followed by normalization of the set using the zscore in built command on MATLAB for the individual vibration signal. The individual extracted feature sets was combined, representing a single set of features labelled as X with 7855 samples and 30 features in total. This was done using the MATLAB code depicted below.  

```matlab

Normalized Dataset
featuresA = statsnormal(A,796,577);
featuresB = statsnormal(B,796,577);

Combine the feature data A and B:
% renaming the variables
featuresX = renamevars(featuresA, featuresA.Properties.VariableNames, strcat(featuresA.Properties.VariableNames, '_X'));
featuresY = renamevars(featuresB, featuresB.Properties.VariableNames, strcat(featuresB.Properties.VariableNames, '_Y'));
% combining the x and y signals
featureset = [featuresX featuresY]
```

## Week 8 Update

After the normalization, the combined features was put into a table and converted into an array for dimensionality reduction purposes. Namely; PCA and CCA reduction techniques have been applied with the following MATLAB code depicted below for visualizations of the results. The correlation graph was also plotted to show the relationship between the true class and predicted class of the 30 features 

```matlab

Table format:
X = table2array(featureset);

PCA:
[coeff, score, latent, tsquared, explained, mu] = pca(X);
figure;
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('Pareto Chart of PCA')

Scatter Plot:
figure;
scatter(score(:,1), score(:,2)); % Scatter plot using the first two principal components
xlabel('PC1');
ylabel('PC2');
title('PCA Scatter Plot');
grid on;

Scatter3 Plot:
figure;
scatter3(score(:,1), score(:,2), score(:,3)); % 3D scatter plot using the first three principal components
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('3D PCA Scatter Plot');
grid on;

Biplot:
figure;
biplot(coeff(:,1:2), 'Scores', score(:,1:2)); % Using first two principal components
xlabel('PC1');
ylabel('PC2');
title('PCA Biplot');
grid on;

Correlation graph:
C = corrcoef(X);
figure;
H = heatmap(C)
% Set the colormap
colormap(H, 'parula');

% titles and labels
H.Title = 'Correlation Heatmap';
H.XLabel = 'Features';
H.YLabel = 'Features';

Reduced dimension:
% intrinsic dimensionality deduced by PCA
figure;
P = score(:,1:2); % reducing dimension via PCA
K = 2; % reduced dimension
lambda0 = 0; % not required for PCA, so put it 0
originaldatadist = squareform(pdist(X));
% dydxplot
dydxplot(P,originaldatadist,K,lambda0);

CCA:
D = X;
K = 8;
epochs = 50;
dist = squareform(pdist(X));
alpha0 = 0.5;
lambda0 = 30;
P = cca(D, K, epochs, [dist],[alpha0],[lambda0]);

figure;
scatter(P(:,1), P(:,2), 'filled');
title('CCA Reduced Embedding');

figure;
scatter(P(:,1), P(:,2), 'filled');
title('CCA Reduced Embedding');

figure;
dydxplot(P,squareform(pdist(X)),K,lambda0)
```
<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/newpca.png" alt="X Statistical Features" title="Principal Component Analysis of the Featureset" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

Depicted in the above figure is the pareto chart after applying the PCA on the final featureset. The dimension has been reduced to 2 principal components with a total percent varaince explained of 99.6254%. 

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/correlation-X.png" alt="Correlation of features" title="Correlation of the Features" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

Illustrated above is the heatmap with the colour indication 
## Week 9 Update
Depicted in the figures bbelow are the scatter plots for the feature set. 

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/scatter1.png" alt="Scatter plot of Mean vs SRM" title="Scatter plot of Mean vs SRM of x signal" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

The figure above depicts the scatter plot for the two feautres mean and srm for the x vibration signal. The scatter plot allows the user to compare any two features with the aid of the dropdown menu on the MATLAB code.

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/scatter2.png" alt="Scatter plot of Mean vs SRM" title="Scatter plot of Mean vs SRM of y signal" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

The figure above depicts the scatter plot for the two feautres mean and srm for the y vibration signal. The scatter plot allows the user to compare any two features with the aid of the dropdown menu on the MATLAB code.

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/scatter3d1.png" alt="Scatter plot of Mean vs Max vs RMS" title="Scatter plot of Mean vs Max vs RMS" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

The above figure illustrates the 3D scatter plot for the three features for the x vibration signal followed by the 3D scatter plot for the same features but for the y vibration signal. 

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/scatter3D2.png" alt="Scatter plot of Mean vs Max vs RMS" title="Scatter plot of Mean vs Max vs RMS" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

## Week 10 Update
Upon attaining the final set of features, having a total of 30 features, the first prediction time (FPT) was computed using MATLAB. This indicates the time at which the degradation process starts or the separating point between healthy and faulty trend. Hence depicted in the two figures below are the determination of FPT value and the trend in the RUL using the FPT value respectively.

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/fpt_featset.png" alt="Determination of FPT value" title="Determination of FPT value" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/fpt_rultrend.png" alt="RUL Trend with FPT" title="RUL Trend with FPT" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

Moreover, model training was carried out beginning with the LSTM model for the whole feature set with the windowing technique.

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/lstm_model.png" alt="LSTM Model for the original featureset" ti tle="LSTM Model for the original featureset" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

The above figure illustartes the trained LSTM model for the raw featureset. The blue  dotted line indicates the actual RUL trend whereas the solid red line indicates the predicted RUL trend. The RMSE value was also computed and resulted in a value of 43.4671.

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/train&test.png" alt="LSTM Model for the divided set" ti tle="LSTM Model for the divided set" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
</figure>

The figure above illustrates the trained LSTM  model for the divided set. The training set was 70% and the test set 30%, however a higher RMSE value was attained.
