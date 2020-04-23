function preprocess_images_balanced(paths, gt_paths, gt_value, dataset_name, save_images)
    path_list = split(paths,';');
    if save_images == true
        endout = regexp(path_list(1), filesep, 'split');
        if endout(1) == ""
            result_images_path = filesep;
        else
            result_images_path = endout(1);
        end
        for level = 2:numel(endout)-2
            result_images_path = fullfile(result_images_path, endout(level));
        end
        result_images_path = fullfile(result_images_path, "matlab_images");
        mkdir(result_images_path);   
    end
    
    gt_path_list = split(gt_paths,';');
    data = [];
    labels = [];
    pick_maps = [];

    for path_idx = 1:length(path_list)
        img = imread(path_list(path_idx));
        gt = imread(gt_path_list(path_idx));
        disp('Current image: ' + path_list(path_idx));
        
        tic
        features = extract_features(img, [3, 5, 10], [3, 5, 10], [50, 50]);
        if save_images == true
           split_path = regexp(path_list(path_idx), filesep, 'split');
           file_name = split_path(end);
           folder_name = strrep(strrep(file_name, '.png', ""), '.jpg', "");
           if endsWith(file_name, ".jpg")
               file_type = ".jpg";
           elseif endsWith(file_name, ".png")
               file_type = ".png";
           end
           mkdir(fullfile(result_images_path, folder_name));
           
           for feature = 1:length(features{2})
               imwrite(features{1}(:,:,feature), fullfile(result_images_path, folder_name, strcat(features{2}(feature), file_type)));
           end
        end
        toc
        images = features{1};
        processed_size = size(images);
        
        gt_pixels = find(gt == gt_value);
        bg_pixels = find(gt == (255 - gt_value));
        n_gt_pixels = numel(gt_pixels);
        n_bg_pixels = numel(bg_pixels);
        n_rand_bg_pixels = min(n_gt_pixels, n_bg_pixels);
        rand_ind = randperm(n_bg_pixels);
        bg_pixels = bg_pixels(rand_ind(1:n_rand_bg_pixels));
        [row_gt, col_gt] = ind2sub(size(gt), gt_pixels);
        [row_bg, col_bg] = ind2sub(size(gt), bg_pixels);
        
        new_data = zeros(n_gt_pixels + n_rand_bg_pixels, processed_size(3), 'single');
        pick_locations = zeros(n_gt_pixels + n_rand_bg_pixels, 3, 'int16');
        for gt_idx = 1:n_gt_pixels
            row = row_gt(gt_idx);
            col = col_gt(gt_idx);
            pick_locations(gt_idx, :) = [path_idx-1 row-1 col-1];
            new_data(gt_idx, :) = images(row, col, :); 
        end
        for bg_idx = 1:n_rand_bg_pixels
            row = row_bg(bg_idx);
            col = col_bg(bg_idx);
            pick_locations(n_gt_pixels + bg_idx, :) = [path_idx-1 row-1 col-1];
            new_data(n_gt_pixels + bg_idx, :) = images(row, col, :); 
        end
        
        data = [data; new_data];
        labels = [labels; ones(n_gt_pixels, 1, 'uint8'); zeros(n_rand_bg_pixels, 1, 'uint8')];
        pick_maps = [pick_maps; pick_locations];
    end
    
    save(strcat(dataset_name, "_balanced.mat"), "data");
    save(strcat(dataset_name, "_balanced_labels.mat"), "labels");
    feature_names = char(features{2});
    save(strcat(dataset_name, "_balanced_feature_names.mat"), "feature_names");
    save(strcat(dataset_name, "_balanced_pick_maps.mat"), "pick_maps");
end