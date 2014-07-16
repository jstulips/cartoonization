%EDGEMAP edge contouring using different types of edge detectors.
function G = edgemap(B, method, max_gradient)

switch(method)
    case 'gradient'             % gradient edge map
        B = rgb2gray(B);
        [GX,GY] = gradient(B(:,:,1));
        G = sqrt(GX.^2+GY.^2);
        G(G>max_gradient) = max_gradient; 
        G = G/max_gradient;
    case 'dizenzo'    % Di Zenzo's multi-spectral gradient
        G = colgrad(B);
        G(isnan(G))=0;          % set NaN values to 0
    case 'sobel'
        %B = double(im2uint8(rgb2gray(B)));
        B = im2double(rgb2gray(B));
        G_x = imfilter(B, fspecial('sobel')','replicate');
        G_y = imfilter(B, fspecial('sobel'),'replicate');
        G_x = double(G_x); 
        G_y = double(G_y);
        G = sqrt(G_x.^2+G_y.^2);
        %G = 1-G;
    case 'canny'
        B = rgb2gray(B);
        G = edge(B,'canny');
        SE = strel('square',2);
        G = imdilate(G,SE);
    case 'log'
        B = rgb2gray(B);
        G = edge(B,'log',[],1.5);         
    case 'dog'   
        B = im2uint8(B);
        G = DOG_Filter(B, 17, 2, 0.75);        % window size, sigma1, sigma2 (where sigma1 > sigma2)
    case 'fdog' 
        %B = rgb2gray(B);
        B = im2double(B);
        G = coherent_line(B);
        G = 1-G;
    otherwise
end