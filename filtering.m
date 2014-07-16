function f = filtering(A,filtermethod)

switch (filtermethod)
    case 'bilateral'
        w     = 5;         % bilateral filter half-width
        sigma = [3 0.0425];   % bilateral filter standard deviations
        % set sigma_r = 4.25% (good value used by Kyprianidis et al.)
        % seems better than sigma_r = 0.1
        f = bfilter2(A,w,sigma);
    case 'kuwahara'
        f = kuwahara_filter(A);
        f = imfilter(f, fspecial('gaussian',[3 3]));   % apply smoothing on filtered image         % smoothen final output
    case 'snn'
        w = 5;                % filter window size
        A_Lab = colorconversion(A,'lab');
        [A_Lab(:,:,1), Apad] = snn(A_Lab(:,:,1), w, true);      % perform on luminance channel only
        f = colorspace('RGB<-Lab',A_Lab);
        % note: this is not needed if anisotropic diffusion is applied
        %f = imfilter(f, fspecial('gaussian',[3 3]));            % smoothen final output
    otherwise
end

    