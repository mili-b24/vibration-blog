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
  <figcaption style="font-style: italic; font-size: 0.9em;">X-Vibration Signal</figcaption>
</figure>

Depcited above is the plot for the x acceleration vibration signal for all the bearing conditions from healthy state to the faulty state.
It can be seen that the amplitude of the signal increases after a few points in time indicating the change in condition of the bearing.

<figure style="text-align: center; margin: 0 auto;">
  <img src="/images/y-vib-signal.png" alt="Y-Vibration Signal" title="Y-Vibration Signal" style="display: block; margin: 0 auto; max-width: 100%; height: auto;" />
  <figcaption style="font-style: italic; font-size: 0.9em;">Y-Vibration Signal</figcaption>
</figure>

Illustrated above is the y acceleration vibration signal for all the conditions of the bearing from a healthy state to the faulty state.
The y vibration signal also shows an increase in amplitude until failure. 

## Week 7 Update

Moreover, we used a function file shown below to extract 15 features from both the x and y vibrations signals:

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

This was followed by normalization of the set of features for both the x and y vibrations signals and combined for further analysis.

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

After the normalization, the combined features was put into a table and converted into an array for dimensionality reduction purposes. Namely; PCA and CCA reduction techniques have been applied with the following MATLAB code depicted below for visualizations of the results. 

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
## Week 9 Update
