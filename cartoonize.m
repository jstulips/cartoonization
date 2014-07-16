function C = cartoonize(A, options)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CARTOONIZE Image abstraction using bilateral filtering.        %
% This function uses the bilateral filter to abstract            %
% an image following the method outlined in:                     %
%                                                                %
% Winnemöller, H., Olsen, S. C., & Gooch, B. (2006).             %
% Real time video abstraction. ACMT Transactions on Graphics,    %
% Proceedings of the ACM SIGGRAPH conference, 25(3), 1221–1226.  %                               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Set image abstraction paramters.
max_gradient      = 0.1;    % maximum gradient (for edges)
sharpness_levels  = [3 14]; % soft quantization sharpness
quant_levels      = 8;      % number of quantization levels
min_edge_strength = 0.2;    % minimum gradient (for edges)

% Set method parameters
cspace = options.cspace;
edgemethod = options.edgemethod;
filtermethod = options.filtermethod;
if isfield(options,'filteredimage')
    B = options.filteredimage;
    disp('Load pre-filtered image');
end

% Apply filtering to input image, if not applied yet
if ~exist('B')
    B = A;
    for i=1:options.nF
        B = filtering(B,filtermethod);
        disp(['Filtering iteration #',num2str(i)]);
        
        if (options.nA == i & strcmp(filtermethod,'snn') )        % when nF == nA
            
            % Perform anisotropic diffusion on luminance channel
            %   for selected filters (SNN...)
            B_Lab = colorconversion(B,cspace);
            B_Lab(:,:,1) = anisodiff2D( B_Lab(:,:,1), 2, 1/7, 30, 1 );    % perform anisotropic diffusion
            B = colorspace('RGB<-Lab',B_Lab); 
            disp('...Anisotropic diffusion');
            clear B_Lab;
       
        end
            
        if (options.nE == i)        % when nF == nE

            % Edge contouring
            G = edgemap(B, edgemethod, max_gradient);
            E = G; 
            %[thresholding disabled]
            %E = (E-min(min(E)))/ (max(max(E)-min(min(E))));  % normalization
            %E = im2bw(E,graythresh(E));
            %E(E<min_edge_strength) = 0;  % limiting gray values to a range (use for
            % gradient edge detector)
            figure, imshow(1-E); title('Edge map');
            disp('...Edge contouring');
        end
    disp('Filtering done');
    %save bf4_man B;        % save a copy if needed [disabled]
    end
end
figure, imshow(B); title(['Filtered image after ',num2str(options.nF),' iterations']);

% Save filtered image [disabled]
%imwrite(B,'filteredImg_snn_test.png','png');

% Apply soft luminance quantization.
B1 = colorconversion(B,cspace);
qB = B1; 
    % Determine per-pixel "sharpening" parameter.
S = diff(sharpness_levels)*G+sharpness_levels(1);

dq = 100/(quant_levels-1);
qB(:,:,1) = (1/dq)*qB(:,:,1);
qB(:,:,1) = dq*round(qB(:,:,1));
qB(:,:,1) = qB(:,:,1)+(dq/2)*tanh(S.*(B1(:,:,1)-qB(:,:,1))); % for temporal coherency
Q = colorspace('RGB<-Lab',qB);
figure; imshow(Q); title('Luminance/color quantized image');

% Combine edges to color-quantized image.
C = repmat(1-E,[1 1 3]) .* Q;