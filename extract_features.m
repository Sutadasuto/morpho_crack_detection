function features = extract_features(img_path, circle_diameters, line_lengths, path_lengths, window_sizes)
    img = imread(img_path);
        
    if length(size(img)) == 3
        img = rgb2gray(img);
    end
    img = imadjust(img,[double(min(img(:)))/255 double(max(img(:)))/255]);
    feature_names = [];
    features = [];
    
    addpath('./frangi_filter_version2a/');
    vessel = FrangiFilter2D(single(img), struct('verbose', false));
    feature_names = [feature_names; "Frangi's vesselness"];
    features = cat(3, features, vessel);
    
    for i = 1:length(line_lengths)
        bottom_hat = bottom_hat_transform(img, line_lengths(i), 10);
        feature_names = [feature_names; "Bottom-hat SE size " + line_lengths(i)];
        features = cat(3, features, bottom_hat);
    end
    
    for i = 1:length(line_lengths)
        cross_bottom_hat = cross_bottom_hat_transform(img, line_lengths(i), 10);
        feature_names = [feature_names; "Cross bottom-hat SE size " + line_lengths(i)];
        features = cat(3, features, cross_bottom_hat);
    end
    
    
    img = im2single(img);
    
    [disks, parameters] = create_circles(circle_diameters);
    [disks, parameters] = delete_duplicates(disks{1}, parameters{1});
    disks = disks{1};
    parameters = parameters{1};
    
    disk_dilations = [];
    for idx = 1:length(disks)
       disk_dilations = cat(3, disk_dilations, imdilate(img, disks(idx))); 
       feature_names = [feature_names; "Dilation " + parameters(idx)];
    end
    features = cat(3, features, disk_dilations);
    
    disk_erosions= [];
    for idx = 1:length(disks)
       disk_erosions = cat(3, disk_erosions, imerode(img, disks(idx)));
       feature_names = [feature_names; "Erosion " + parameters(idx)];
    end
    features = cat(3, features, disk_erosions);
    
    disk_closings = [];
    for idx = 1:length(disks)
       disk_closings = cat(3, disk_closings, imclose(img, disks(idx)));
       feature_names = [feature_names; "Closing " + parameters(idx)];
    end
    features = cat(3, features, disk_closings);
    
    disk_openings = [];
    for idx = 1:length(disks)
       disk_openings = cat(3, disk_openings, imopen(img, disks(idx)));
       feature_names = [feature_names; "Opening " + parameters(idx)];
    end
    features = cat(3, features, disk_openings);
    
    line_closings = [];
    for i = 1:length(line_lengths)
        [lines, parameters] = create_oriented_lines(line_lengths(i), 2);
        lines = lines{1};
        parameters = parameters{1};
        line_closings = cat(3, line_closings, multi_oriented_close(img, lines));
        feature_names = [feature_names; "Closing Multi-dir Line length " + line_lengths(i)];
    end
    features = cat(3, features, line_closings);
    
    command = strcat("python smil.py '", img_path, "' --path_sizes ", strjoin(string(path_lengths)));
    [status,cmdout] = system(command);
    for i = 1:length(path_lengths)
        features = cat(3, features, im2single(imread(strcat("matlab_closing_", string(path_lengths(i)), ".png"))));
        feature_names = [feature_names; "Path Closing length " + path_lengths(i)];
        delete(strcat("matlab_closing_", string(path_lengths(i)), ".png"));
    end
    
    for i = 1:length(window_sizes)
        window_size = window_sizes{i};
        sliding_mean = sliding_statistical_measure(img, window_size, "mean");
        feature_names = [feature_names; "Sliding mean " + string(window_size(1)) + "x" + string(window_size(2))];
        features = cat(3, features, sliding_mean);
        sliding_median = sliding_statistical_measure(img, window_size, "median");
        feature_names = [feature_names; "Sliding median " + string(window_size(1)) + "x" + string(window_size(2))];
        features = cat(3, features, sliding_median);
        sliding_std = sliding_statistical_measure(img, window_size, "std");
        feature_names = [feature_names; "Sliding std " + string(window_size(1)) + "x" + string(window_size(2))];
        features = cat(3, features, sliding_std);
        sliding_mad = sliding_statistical_measure(img, window_size, "mad");
        feature_names = [feature_names; "Sliding mad " + string(window_size(1)) + "x" + string(window_size(2))];
        features = cat(3, features, sliding_mad);
    end
    features = {features, feature_names};
end

function [structuring_elements, parameters] = create_circles(diameters)
    radii = floor(diameters/2);
    structuring_elements = [strel('disk', radii(1))];
    parameters = ["Disk diameter " + num2str(diameters(1))];
    for idx = 2:length(radii)
        new_se = strel('disk', radii(idx));
        if ~(new_se == structuring_elements(end))
            structuring_elements(end+1) = new_se;
            parameters(end+1) = "Disk diameter " + num2str(diameters(idx));
        end
    end
    structuring_elements = {structuring_elements};
    parameters = {parameters};
end

function [structuring_elements, parameters] = create_oriented_lines(line_size, dir_step)
    orientations = 0:dir_step:179;
    structuring_elements = [strel('line', line_size, orientations(1))];
    parameters = ["Line length " + num2str(line_size) + " orientation " + num2str(orientations(1))];
    for idx = 2:length(orientations)
        new_se = strel('line', line_size, orientations(idx));
        if ~(new_se == structuring_elements(end))
            structuring_elements(end+1) = new_se;
            parameters(end+1) = ["Line length " + num2str(line_size) + " orientation " + num2str(orientations(idx))];
        end
    end
    structuring_elements = {structuring_elements};
    parameters= {parameters};
end

function [structuring_elements, parameters] = delete_duplicates(se_array, parameters_array)
    structuring_elements = [se_array(1)];
    parameters = [parameters_array(1)];
    for idx = 2:length(se_array)
        already_exists = false;
        for idx2 = length(structuring_elements):-1:1
            if se_array(idx) == structuring_elements(idx2)
                already_exists = true;
            end
        end
        if ~already_exists
            structuring_elements(end+1) = se_array(idx);
            parameters(end+1) = parameters_array(idx);
        end
    end
    structuring_elements = {structuring_elements};
    parameters = {parameters};
end

function result = sliding_statistical_measure(img, window_size, measure)
    if mod(window_size(1), 2) == 0
        window_size(1) = window_size(1) + 1;
    end
    if mod(window_size(2), 2) == 0
        window_size(2) = window_size(2) + 1;
    end
%     img_size = size(img);
    kernel = true(window_size);
    
    switch measure
       case "mean"
           result = imfilter(img, double(kernel), 'symmetric')/(window_size(1)*window_size(2));
       case "std"
           result = stdfilt(img, kernel);
       case "median"
           result = medfilt2(img, window_size, 'symmetric');
       case "mad"
           result = 1.4826 * medfilt2(abs(img - medfilt2(img, window_size, "symmetric")), window_size, "symmetric");
%            medians = medfilt2(img, window_size, 'symmetric');
%            v_wing = floor(window_size(1)/2);
%            h_wing = floor(window_size(2)/2);
%            result = zeros(size(img));
%            
%            for i = 1:img_size(1)
%                for j = 1:img_size(2)
%                    min_y = max(1, i-v_wing);
%                    max_y = min(img_size(1), i+v_wing);
%                    min_x = max(1, j-h_wing);
%                    max_x = min(img_size(2), j+h_wing);
%                    window = abs(img(min_y:max_y, min_x:max_x) - medians(i,j));
%                    result(i, j) = median(window, "all");
%                end
%            end
    end   
end

function closing = multi_oriented_close(gray_image, structuring_elements)
    image_size = size(gray_image);
    closings = zeros(image_size(1), image_size(2), length(structuring_elements), 'single');
    
    for idx = 1:length(structuring_elements)
        closings(:, :, idx) = imclose(gray_image, structuring_elements(idx));
    end

    infimum = closings(:, :, 1);
    for idx = 2:length(structuring_elements)
        infimum = min(infimum, closings(:, :, idx));
    end
    closing = im2single(infimum);
end

function bottom_hat = bottom_hat_transform(gray_image, line_size, dir_step)
    [structuring_elements, parameters] = create_oriented_lines(line_size, dir_step);
    [structuring_elements, parameters] = delete_duplicates(structuring_elements{1}, parameters{1});
    structuring_elements = structuring_elements{1};
    image_size = size(gray_image);

    bottom_hats = zeros(image_size(1), image_size(2), length(structuring_elements), 'uint8');
    for idx = 1:length(structuring_elements)
        bottom_hats(:, :, idx) = imbothat(gray_image, structuring_elements(idx));
    end

    supremum = bottom_hats(:, :, 1);
    for idx = 2:length(structuring_elements)
        supremum = max(supremum, bottom_hats(:, :, idx));
    end
    bottom_hat = im2single(supremum);
end

function cross_bottom_hat = cross_bottom_hat_transform(gray_image, line_size, dir_step)
angle = 0;
supremum = zeros(size(gray_image), "uint8");
while angle < 180
    se = strel('line', line_size, angle);
    se_2 = strel('line', 2*line_size, angle + 90);
    bh = imbothat(imopen(gray_image, se), se);
    bh_2 = imbothat(imopen(gray_image, se), se_2);
    cross_bh = uint8(max(zeros(size(gray_image), "int16"), int16(bh) - int16(bh_2)));
    supremum = max(supremum, cross_bh);
    angle = angle + dir_step;
end
cross_bottom_hat = im2single(supremum);
end