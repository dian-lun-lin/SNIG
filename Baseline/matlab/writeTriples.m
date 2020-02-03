function writeTriples(A,fname);
% Write out non-entries of matrix A as triples in a TSV file.

  % Initialized triples matrix.
  ijv = zeros(3,nnz(A));
  
  % Get the non-zero entries.
  [ijv(1,:) ijv(2,:) ijv(3,:)] = find(A);
  
  % Open file for write.
  fid=fopen(fname,'w');
  
  % Write data.
  fprintf(fid,'%d\t%d\t%d\n',ijv);
  
  % Close file.
  fclose(fid);

return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Software Engineer: Dr. Jeremy Kepner                    
% MIT                   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) <2019> Massachusetts Institute of Technology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

