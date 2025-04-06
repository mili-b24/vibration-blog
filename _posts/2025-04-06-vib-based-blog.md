---
layout: post
title: Vibrations-based data exploration for induction motor bearing
---

Rotating machines has become a crucial component for many industrial processes today and failure of this component has a great impact on the overall performance of industries.
Upon attaining the vibrations signal of the bearing in the x and y directions, the data was explored using the MATLAB software and the code given below. The vibrations signal were attained as mlx files and has
to be loaded first into MATLAB's workspace for further exploration of the data. The two signals are plotted against time as well for further visualization labelled as timestamps. The timestamps was also analyzed 
in order to convert it to datetime format for easier labelling of data collection allocation for which particular day and hour. 

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
