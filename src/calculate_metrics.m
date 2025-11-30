function [p, s, e] = calculate_metrics(clean, denoised)
    % PSNR
    p = psnr(denoised, clean);
    
    % SSIM
    s = ssim(denoised, clean);
    
    % EPI (Edge Preservation Index)
    % Gradient calculation
    [Gx1, Gy1] = imgradientxy(clean);
    [Gx2, Gy2] = imgradientxy(denoised);
    
    grad1 = sqrt(Gx1.^2 + Gy1.^2);
    grad2 = sqrt(Gx2.^2 + Gy2.^2);
    
    % Correlation
    num = sum(sum(grad1 .* grad2));
    den = sqrt(sum(sum(grad1.^2)) * sum(sum(grad2.^2)));
    e = num / (den + 1e-8);
end