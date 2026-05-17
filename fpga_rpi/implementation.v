//////////////////////////////////////////////////////////////////////////////
// Module: rpi5_gpio_controller
// Description: SPI-like slave controller for Raspberry Pi 5 GPIO interface
//////////////////////////////////////////////////////////////////////////////

module rpi5_gpio_controller #(
    parameter CLK_FREQ_MHZ = 100,
    parameter SPI_MAX_MHZ = 10
)(
    input  wire        sys_clk,
    input  wire        sys_rst_n,
    
    // Physical GPIO Interface
    input  wire        gpio_sclk,
    input  wire        gpio_mosi,
    output reg         gpio_miso,
    input  wire        gpio_cs_n,
    output wire        gpio_irq,
    
    // Crossbar Array Interface
    output reg  [7:0]  cb_chip_sel,
    output reg  [15:0] cb_row_addr,
    output reg  [15:0] cb_col_addr,
    output reg  [31:0] cb_wdata,
    input  wire [31:0] cb_rdata,
    output reg         cb_wr_en,
    output reg         cb_rd_en,
    input  wire        cb_busy,
    output reg         cb_start,
    input  wire        cb_done,
    
    // SADP Controller Interface
    output reg         sadp_start,
    output reg  [15:0] sadp_target_row,
    output reg  [3:0]  sadp_target_col,
    input  wire        sadp_busy,
    input  wire        sadp_done,
    input  wire [1:0]  sadp_status,
    
    // Encoder/Decoder Pipeline Interface
    output reg         enc_start,
    output reg         dec_start,
    input  wire        enc_busy,
    input  wire        dec_busy,
    
    // Status Registers
    output reg  [31:0] status_reg,
    input  wire [31:0] error_reg
);

    // Synchronization registers for async inputs
    reg [2:0] sclk_sync, mosi_sync, cs_sync;
    reg [1:0] sclk_edge_det;
    
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            sclk_sync <= 3'b0;
            mosi_sync <= 3'b0;
            cs_sync   <= 3'b1; // CS inactive (high)
        end else begin
            sclk_sync <= {sclk_sync[1:0], gpio_sclk};
            mosi_sync <= {mosi_sync[1:0], gpio_mosi};
            cs_sync   <= {cs_sync[1:0], gpio_cs_n};
        end
    end
    
    // Edge detection
    always @(posedge sys_clk) begin
        sclk_edge_det <= {sclk_edge_det[0], sclk_sync[2]};
    end
    
    wire sclk_rising  = (sclk_edge_det == 2'b01);
    wire sclk_falling = (sclk_edge_det == 2'b10);
    wire cs_active    = !cs_sync[2];
    wire cs_rising    = (cs_sync[2:1] == 2'b01);
    
    // SPI State Machine
    localparam [3:0] ST_IDLE        = 4'd0,
                     ST_SYNC_H      = 4'd1,
                     ST_SYNC_L      = 4'd2,
                     ST_CMD         = 4'd3,
                     ST_ADDR_CHIP   = 4'd4,
                     ST_ADDR_ROW_H  = 4'd5,
                     ST_ADDR_ROW_L  = 4'd6,
                     ST_ADDR_COL_H  = 4'd7,
                     ST_ADDR_COL_L  = 4'd8,
                     ST_DATA_3      = 4'd9,
                     ST_DATA_2      = 4'd10,
                     ST_DATA_1      = 4'd11,
                     ST_DATA_0      = 4'd12,
                     ST_CRC         = 4'd13,
                     ST_EXECUTE     = 4'd14,
                     ST_RESPOND     = 4'd15;
    
    reg [3:0]  state, next_state;
    reg [2:0]  bit_cnt;
    reg [7:0]  shift_reg;
    reg [7:0]  rx_crc, calc_crc;
    reg [7:0]  cmd_reg;
    reg [7:0]  resp_buf [0:7];
    reg [2:0]  resp_cnt;
    reg [2:0]  resp_bit_cnt;
    
    // Shift register for RX
    always @(posedge sys_clk) begin
        if (cs_active && sclk_rising) begin
            shift_reg <= {shift_reg[6:0], mosi_sync[2]};
            bit_cnt <= bit_cnt + 1'b1;
        end else if (!cs_active) begin
            bit_cnt <= 3'd0;
        end
    end
    
    wire byte_done = (bit_cnt == 3'd7) && sclk_rising;
    
    // CRC calculation (XOR checksum)
    always @(posedge sys_clk) begin
        if (!sys_rst_n) begin
            calc_crc <= 8'h00;
        end else if (cs_rising) begin
            calc_crc <= 8'h00;
        end else if (byte_done && state != ST_CRC) begin
            calc_crc <= calc_crc ^ shift_reg;
        end
    end
    
    // State machine - sequential
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            state <= ST_IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    // State machine - combinational
    always @(*) begin
        next_state = state;
        case (state)
            ST_IDLE:       if (cs_active) next_state = ST_SYNC_H;
            ST_SYNC_H:     if (byte_done && shift_reg == 8'hAA) next_state = ST_SYNC_L;
                           else if (byte_done) next_state = ST_IDLE;
            ST_SYNC_L:     if (byte_done && shift_reg == 8'h55) next_state = ST_CMD;
                           else if (byte_done) next_state = ST_IDLE;
            ST_CMD:        if (byte_done) next_state = ST_ADDR_CHIP;
            ST_ADDR_CHIP:  if (byte_done) next_state = ST_ADDR_ROW_H;
            ST_ADDR_ROW_H: if (byte_done) next_state = ST_ADDR_ROW_L;
            ST_ADDR_ROW_L: if (byte_done) next_state = ST_ADDR_COL_H;
            ST_ADDR_COL_H: if (byte_done) next_state = ST_ADDR_COL_L;
            ST_ADDR_COL_L: if (byte_done) next_state = ST_DATA_3;
            ST_DATA_3:     if (byte_done) next_state = ST_DATA_2;
            ST_DATA_2:     if (byte_done) next_state = ST_DATA_1;
            ST_DATA_1:     if (byte_done) next_state = ST_DATA_0;
            ST_DATA_0:     if (byte_done) next_state = ST_CRC;
            ST_CRC:        if (byte_done) next_state = (shift_reg == calc_crc) ? ST_EXECUTE : ST_IDLE;
            ST_EXECUTE:    next_state = ST_RESPOND;
            ST_RESPOND:    if (resp_done) next_state = ST_IDLE;
            default:       next_state = ST_IDLE;
        endcase
    end
    
    // Byte capture registers
    always @(posedge sys_clk) begin
        if (byte_done) begin
            case (state)
                ST_CMD:        cmd_reg <= shift_reg;
                ST_ADDR_CHIP:  cb_chip_sel <= shift_reg;
                ST_ADDR_ROW_H: cb_row_addr[15:8] <= shift_reg;
                ST_ADDR_ROW_L: cb_row_addr[7:0]  <= shift_reg;
                ST_ADDR_COL_H: cb_col_addr[15:8] <= shift_reg;
                ST_ADDR_COL_L: cb_col_addr[7:0]  <= shift_reg;
                ST_DATA_3:     cb_wdata[31:24]   <= shift_reg;
                ST_DATA_2:     cb_wdata[23:16]   <= shift_reg;
                ST_DATA_1:     cb_wdata[15:8]    <= shift_reg;
                ST_DATA_0:     cb_wdata[7:0]     <= shift_reg;
            endcase
        end
    end
    
    // Command execution
    reg resp_done;
    
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            cb_wr_en   <= 1'b0;
            cb_rd_en   <= 1'b0;
            cb_start   <= 1'b0;
            sadp_start <= 1'b0;
            enc_start  <= 1'b0;
            dec_start  <= 1'b0;
            resp_done  <= 1'b0;
        end else begin
            // Default clear pulses
            cb_wr_en   <= 1'b0;
            cb_rd_en   <= 1'b0;
            cb_start   <= 1'b0;
            sadp_start <= 1'b0;
            enc_start  <= 1'b0;
            dec_start  <= 1'b0;
            resp_done  <= 1'b0;
            
            if (state == ST_EXECUTE) begin
                case (cmd_reg)
                    8'h01: begin // READ_REG
                        cb_rd_en <= 1'b1;
                        resp_buf[0] <= cb_rdata[31:24];
                        resp_buf[1] <= cb_rdata[23:16];
                        resp_buf[2] <= cb_rdata[15:8];
                        resp_buf[3] <= cb_rdata[7:0];
                    end
                    
                    8'h02: begin // WRITE_REG
                        cb_wr_en <= 1'b1;
                        resp_buf[0] <= 8'h00; // ACK
                    end
                    
                    8'h10: begin // SADP_PROGRAM
                        sadp_target_row <= cb_row_addr;
                        sadp_target_col <= cb_col_addr[3:0];
                        sadp_start <= 1'b1;
                        resp_buf[0] <= 8'h10; // SADP started
                    end
                    
                    8'h20: begin // LOAD_WEIGHT
                        cb_start <= 1'b1;
                        resp_buf[0] <= 8'h20; // Load started
                    end
                    
                    8'h30: begin // PREDICT_FWD
                        cb_start <= 1'b1;
                        resp_buf[0] <= 8'h30;
                        // Response will be 16384 bytes, handled by DMA
                    end
                    
                    8'h40: begin // ENCODE_STEGO
                        enc_start <= 1'b1;
                        resp_buf[0] <= 8'h40;
                    end
                    
                    8'h50: begin // DECODE_STEGO
                        dec_start <= 1'b1;
                        resp_buf[0] <= 8'h50;
                    end
                    
                    8'h60: begin // READ_ARRAY_STATUS
                        resp_buf[0] <= status_reg[31:24];
                        resp_buf[1] <= status_reg[23:16];
                        resp_buf[2] <= status_reg[15:8];
                        resp_buf[3] <= status_reg[7:0];
                        resp_buf[4] <= error_reg[31:24];
                        resp_buf[5] <= error_reg[23:16];
                        resp_buf[6] <= error_reg[15:8];
                        resp_buf[7] <= error_reg[7:0];
                    end
                    
                    8'hFF: begin // RESET_FPGA
                        resp_buf[0] <= 8'hFF;
                    end
                    
                    default: resp_buf[0] <= 8'hEE; // Error code
                endcase
            end
            
            if (state == ST_RESPOND && resp_bit_done) begin
                resp_done <= 1'b1;
            end
        end
    end
    
    // Response state machine (MISO drive)
    reg [2:0] resp_byte_idx;
    wire resp_bit_done = (resp_bit_cnt == 3'd7) && sclk_falling;
    
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            gpio_miso <= 1'b0;
            resp_bit_cnt <= 3'd0;
            resp_byte_idx <= 3'd0;
        end else if (!cs_active) begin
            resp_bit_cnt <= 3'd0;
            resp_byte_idx <= 3'd0;
            gpio_miso <= 1'b0;
        end else if (state == ST_RESPOND) begin
            if (sclk_falling) begin
                gpio_miso <= resp_buf[resp_byte_idx][7 - resp_bit_cnt];
                resp_bit_cnt <= resp_bit_cnt + 1'b1;
            end
            
            if (resp_bit_done) begin
                resp_byte_idx <= resp_byte_idx + 1'b1;
            end
        end else begin
            gpio_miso <= 1'b0;
        end
    end
    
    // Status register update
    always @(posedge sys_clk) begin
        status_reg[0] <= cb_busy;
        status_reg[1] <= sadp_busy;
        status_reg[2] <= enc_busy;
        status_reg[3] <= dec_busy;
        status_reg[7:4] <= sadp_status;
        status_reg[31:8] <= 24'h000000;
    end
    
    assign gpio_irq = cb_done || sadp_done;

endmodule