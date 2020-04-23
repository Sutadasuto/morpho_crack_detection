function results = my_test(img, len, ori_step)
angle = 0;
supremum = zeros(size(img), "uint8");
ori_im = img;
while angle < 180
    se = strel('line', len, angle);
    se_2 = strel('line', 2*len, angle + 90);
    bh = imbothat(imopen(img, se), se);
    bh_2 = imbothat(imopen(img, se), se_2);
    cross_bh = uint8(max(zeros(size(img), "int16"), int16(bh) - int16(bh_2)));
    supremum = max(supremum, cross_bh);
    angle = angle + ori_step;
end
cross_bottom_hat = imadjust(supremum);
results = [cat(3, ori_im, ori_im, ori_im) cat(3, cross_bottom_hat, ori_im, ori_im)]; 
imagesc(results)
colormap gray
end

function clean = clean_cross_bh(cross_bottom_hat, len)
    filtered = medfilt2(cross_bottom_hat, [ceil(len/2) ceil(len/2)]);
    skeleton = bwmorph(filtered, "skel", Inf);
    closed = imclose(skeleton, strel("disk", ceil(len/2)));
    measurements = regionprops(closed, "Area", "Eccentricity", "PixelIdxList");
    for i=1:length(measurements)
       if measurements(i).Area < len || measurements(i).Eccentricity < 0.8
          closed(measurements(i).PixelIdxList) = 0; 
       end
    end
    clean = im2uint8(closed);
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
    bottom_hat = supremum;
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