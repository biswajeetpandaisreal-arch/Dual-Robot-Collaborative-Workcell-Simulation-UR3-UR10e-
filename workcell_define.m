function cell = workcell_define()

cell.benchX = 1250;
cell.benchY = 1945;

cell.holder = struct('X',90,'Y',150,'Z',10,'elev',140);
cell.pcb    = struct('X',85.09,'Y',56.134,'Z',5);

cell.Pos3 = [505,  945, 140];
cell.Pos4 = [1055, 405, 140];

cell.P = [
    1016.237, 368.043, 140;
    1002.775, 368.043, 140;
    1002.775, 395.983, 140;
    1002.775, 419.097, 140
];

cell.camera = struct('x',cell.benchX/2,'y',cell.benchY/2,'z',1950);

end
