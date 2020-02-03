function A = readTriples(fname);
% Read triples for a matrix from a TSV file and
% build a sparse matrix from the triples.

  % Read data from file into a triples matrix.
  ijv = transpose(reshape(sscanf(StrFileRead(fname),'%f'),3,[]));
  
  % Create sparse matrix from triplses.
  A = sparse(ijv(:,1),ijv(:,2),ijv(:,3));

return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Software Engineer: Dr. Jeremy Kepner                    
% MIT                   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) <2019> Massachusetts Institute of Technology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

