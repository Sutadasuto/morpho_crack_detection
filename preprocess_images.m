function preprocess_images(paths, dataset_name)
    results_dir = strcat(dataset_name, ".mat");
    mkdir(results_dir);
    path_list = split(paths,';');
    path_list = string(path_list);

    for path_idx = 1:length(path_list)
        split_path = regexp(path_list(path_idx), filesep, 'split');
        file_name = split_path(end);
        img = imread(path_list(path_idx));
        disp('Current image: ' + path_list(path_idx));
        
        tic
        features = extract_features(img, [3, 5, 10], [3, 5, 10], [50, 50]);
        toc
        
        images = features{1};
        resulting_image_path = fullfile(results_dir, strrep(strrep(file_name,'.png','.mat'), '.jpg', '.mat'));
        save(resulting_image_path, "images");
    end
    
    feature_names = char(features{2});
    save("feature_names.mat", "feature_names");
end

function features = extract_features(img, circle_diameters, line_lengths, window_size)
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
    bottom_hat = bottom_hat_transform(img, 10, 10);
    feature_names = [feature_names; "Bottom-hat"];
    features = cat(3, features, bottom_hat);
    
    
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
    
    lines = [];
    line_parameters = [];
    for i = 1:length(line_lengths)
        [new_lines, new_parameters] = create_oriented_lines(line_lengths(i), 2);
        new_lines = new_lines{1};
        new_parameters = new_parameters{1};
        lines = [lines new_lines];
        for j = 1:length(new_lines)
            line_parameters = [line_parameters "Closing " + new_parameters(j)];
        end
    end
    [lines, line_parameters] = delete_duplicates(lines, line_parameters);
    lines = lines{1};
    feature_names = [feature_names; line_parameters{1}.transpose()];
    
    line_closings = [];
    for idx = 1:length(lines)
       line_closings = cat(3, line_closings, imclose(img, lines(idx)));
    end
    features = cat(3, features, line_closings);
    
    sliding_mean = sliding_statistical_measure(img, window_size, "mean");
    feature_names = [feature_names; "Sliding mean"];
    features = cat(3, features, sliding_mean);
    sliding_median = sliding_statistical_measure(img, window_size, "median");
    feature_names = [feature_names; "Sliding median"];
    features = cat(3, features, sliding_median);
    sliding_std = sliding_statistical_measure(img, window_size, "std");
    feature_names = [feature_names; "Sliding std"];
    features = cat(3, features, sliding_std);
%     sliding_mad = sliding_statistical_measure(img, window_size, "mad");
%     feature_names = [feature_names; "Sliding mad"];
%     features = cat(3, features, sliding_mad);
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
    img_size = size(img);
    kernel = true(window_size);
    
    switch measure
       case "mean"
           result = imfilter(img, double(kernel), 'symmetric')/(window_size(1)*window_size(2));
       case "std"
           result = stdfilt(img, kernel);
       case "median"
           result = medfilt2(img, window_size, 'symmetric');
       case "mad"
           medians = medfilt2(img, window_size, 'symmetric');
           v_wing = floor(window_size(1)/2);
           h_wing = floor(window_size(2)/2);
           result = zeros(size(img));
           
           for i = 1:img_size(1)
               for j = 1:img_size(2)
                   min_y = max(1, i-v_wing);
                   max_y = min(img_size(1), i+v_wing);
                   min_x = max(1, j-h_wing);
                   max_x = min(img_size(2), j+h_wing);
                   window = abs(img(min_y:max_y, min_x:max_x) - medians(i,j));
                   result(i, j) = median(window, "all");
               end
           end
    end   
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
