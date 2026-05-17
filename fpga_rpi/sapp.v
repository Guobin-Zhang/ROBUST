//////////////////////////////////////////////////////////////////////////////
// Module: sadp_controller
// Description: Sensitivity-Adaptive Device Programming for memristor calibration
//////////////////////////////////////////////////////////////////////////////

module sadp_controller #(
    parameter T_PROBE_NS = 20,
    parameter CLK_PERIOD_NS = 10,  // 100MHz
    parameter G_MAX_US = 100,      // 100uS max conductance
    parameter G_MIN_US = 1         // 1uS min conductance
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // Control Interface
    input  wire        start,
    output reg         busy,
    output reg         done,
    output reg  [1:0]  status, // 00:success, 01:max_iter, 10:verify_fail, 11:timeout
    
    // Target Configuration
    input  wire [15:0] target_row,
    input  wire [3:0]  target_col,
    input  wire [31:0] target_g, // Target conductance in fixed-point
    
    // Crossbar Array Interface
    output reg  [15:0] cb_row,
    output reg  [15:0] cb_col,
    output reg         cb_prog_en,
    output reg         cb_read_en,
    output reg  [7:0]  cb_prog_voltage,  // 0.4V to 2.0V encoded
    output reg  [15:0] cb_prog_width,    // Pulse width in ns
    output reg         cb_prog_polarity, // 0:positive, 1:negative
    input  wire [31:0] cb_read_data,
    input  wire        cb_read_valid
);

    // Timing constants
    localparam T_PROBE_CYCLES = T_PROBE_NS / CLK_PERIOD_NS;
    localparam SADP_ULTRA_SENSITIVE = 2'd0;
    localparam SADP_STANDARD = 2'd1;
    localparam SADP_ROBUST = 2'd2;
    
    // Programming parameters table
    reg [7:0]  prog_voltage [0:2];
    reg [15:0] prog_width [0:2];
    reg [4:0]  prog_max_iter [0:2];
    
    initial begin
        // Ultra-sensitive: 0.4V, 5ns, 20 iterations
        prog_voltage[SADP_ULTRA_SENSITIVE] = 8'd20;   // 0.4V * 50
        prog_width[SADP_ULTRA_SENSITIVE]   = 16'd5;
        prog_max_iter[SADP_ULTRA_SENSITIVE] = 5'd20;
        
        // Standard: 1.0V, 10ns, 15 iterations
        prog_voltage[SADP_STANDARD] = 8'd50;  // 1.0V * 50
        prog_width[SADP_STANDARD]   = 16'd10;
        prog_max_iter[SADP_STANDARD] = 5'd15;
        
        // Robust: 2.0V, 50ns, 10 iterations
        prog_voltage[SADP_ROBUST] = 8'd100; // 2.0V * 50
        prog_width[SADP_ROBUST]   = 16'd50;
        prog_max_iter[SADP_ROBUST] = 5'd10;
    end
    
    // State machine
    localparam [3:0] ST_IDLE = 4'd0,
                     ST_SENS_TEST_1 = 4'd1,
                     ST_SENS_TEST_2 = 4'd2,
                     ST_SENS_TEST_3 = 4'd3,
                     ST_CLASSIFY = 4'd4,
                     ST_PROG_START = 4'd5,
                     ST_PROG_PULSE = 4'd6,
                     ST_PROG_VERIFY = 4'd7,
                     ST_PROG_ADJUST = 4'd8,
                     ST_COMPENSATION = 4'd9,
                     ST_FINAL_VERIFY = 4'd10,
                     ST_DONE = 4'd11;
    
    reg [3:0] state;
    reg [1:0] sensitivity_class;
    reg [31:0] g_probe [0:2]; // Conductance readings from 3 probe pulses
    reg [31:0] g_current;
    reg [31:0] g_target;
    reg [4:0]  iter_count;
    reg [31:0] timer;
    
    // Sensitivity calculation: S = median(dG) / (V_probe * T_probe)
    wire [31:0] dg1 = (g_probe[0] > g_probe[1]) ? (g_probe[0] - g_probe[1]) : (g_probe[1] - g_probe[0]);
    wire [31:0] dg2 = (g_probe[1] > g_probe[2]) ? (g_probe[1] - g_probe[2]) : (g_probe[2] - g_probe[1]);
    reg [31:0] dg_median;
    
    always @(*) begin
        if ((dg1 <= dg2 && dg1 >= dg2) || (dg1 >= dg2 && dg1 <= dg2))
            dg_median = dg1;
        else if ((dg2 <= dg1 && dg2 >= dg1) || (dg2 >= dg1 && dg2 <= dg1))
            dg_median = dg2;
        else
            dg_median = (g_probe[0] > g_probe[1]) ? (g_probe[0] - g_probe[2]) : (g_probe[2] - g_probe[0]);
    end
    
    // Main state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            status <= 2'b00;
            cb_prog_en <= 1'b0;
            cb_read_en <= 1'b0;
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        state <= ST_SENS_TEST_1;
                        cb_row <= target_row;
                        cb_col <= {12'd0, target_col};
                        g_target <= target_g;
                    end
                end
                
                ST_SENS_TEST_1: begin
                    // Apply 0.8V, 20ns probe pulse
                    cb_prog_voltage <= 8'd40; // 0.8V
                    cb_prog_width <= 16'd20;
                    cb_prog_en <= 1'b1;
                    if (timer >= T_PROBE_CYCLES) begin
                        cb_prog_en <= 1'b0;
                        state <= ST_SENS_TEST_2;
                        timer <= 32'd0;
                    end else begin
                        timer <= timer + 1;
                    end
                end
                
                ST_SENS_TEST_2: begin
                    // Apply 1.2V, 20ns probe pulse
                    cb_prog_voltage <= 8'd60; // 1.2V
                    cb_prog_en <= 1'b1;
                    if (timer >= T_PROBE_CYCLES) begin
                        cb_prog_en <= 1'b0;
                        state <= ST_SENS_TEST_3;
                        timer <= 32'd0;
                    end else begin
                        timer <= timer + 1;
                    end
                end
                
                ST_SENS_TEST_3: begin
                    // Apply 1.6V, 20ns probe pulse
                    cb_prog_voltage <= 8'd80; // 1.6V
                    cb_prog_en <= 1'b1;
                    if (timer >= T_PROBE_CYCLES) begin
                        cb_prog_en <= 1'b0;
                        state <= ST_CLASSIFY;
                        timer <= 32'd0;
                    end else begin
                        timer <= timer + 1;
                    end
                end
                
                ST_CLASSIFY: begin
                    // Classify based on sensitivity S
                    if (dg_median < 32'd30) // S < 0.3
                        sensitivity_class <= SADP_ULTRA_SENSITIVE;
                    else if (dg_median > 32'd150) // S > 1.5
                        sensitivity_class <= SADP_ROBUST;
                    else
                        sensitivity_class <= SADP_STANDARD;
                    
                    iter_count <= 5'd0;
                    state <= ST_PROG_START;
                end
                
                ST_PROG_START: begin
                    // Setup programming parameters based on classification
                    cb_prog_voltage <= prog_voltage[sensitivity_class];
                    cb_prog_width <= prog_width[sensitivity_class];
                    state <= ST_PROG_PULSE;
                end
                
                ST_PROG_PULSE: begin
                    // Apply programming pulse
                    cb_prog_en <= 1'b1;
                    if (timer >= cb_prog_width / CLK_PERIOD_NS) begin
                        cb_prog_en <= 1'b0;
                        state <= ST_PROG_VERIFY;
                        timer <= 32'd0;
                    end else begin
                        timer <= timer + 1;
                    end
                end
                
                ST_PROG_VERIFY: begin
                    // Read current conductance
                    cb_read_en <= 1'b1;
                    if (cb_read_valid) begin
                        cb_read_en <= 1'b0;
                        g_current <= cb_read_data;
                        
                        // Check convergence: |G - G_target| / G_range < 0.78%
                        if ((g_current > g_target ? (g_current - g_target) : (g_target - g_current)) 
                            < ((G_MAX_US - G_MIN_US) * 78 / 10000)) begin
                            state <= ST_FINAL_VERIFY;
                        end else if (iter_count >= prog_max_iter[sensitivity_class]) begin
                            status <= 2'b01; // Max iteration reached
                            state <= ST_DONE;
                        end else begin
                            state <= ST_PROG_ADJUST;
                        end
                    end
                end
                
                ST_PROG_ADJUST: begin
                    // Adjust programming parameters based on error
                    iter_count <= iter_count + 1;
                    
                    if (g_current > g_target && g_current - g_target > (G_MAX_US - G_MIN_US) / 10) begin
                        // Overshoot, apply compensation pulse
                        cb_prog_polarity <= 1'b1; // Negative
                        cb_prog_voltage <= prog_voltage[sensitivity_class] / 3;
                        cb_prog_width <= prog_width[sensitivity_class] / 2;
                        state <= ST_COMPENSATION;
                    end else begin
                        // Normal programming, adjust step size
                        if ((g_current > g_target ? (g_current - g_target) : (g_target - g_current)) 
                            > (G_MAX_US - G_MIN_US) / 10) begin
                            // Large error, keep parameters
                        end else if ((g_current > g_target ? (g_current - g_target) : (g_target - g_current)) 
                                     > (G_MAX_US - G_MIN_US) / 20) begin
                            // Medium error, reduce by half
                            cb_prog_voltage <= prog_voltage[sensitivity_class] / 2;
                            cb_prog_width <= prog_width[sensitivity_class] / 2;
                        end else begin
                            // Small error, fine mode
                            cb_prog_voltage <= prog_voltage[sensitivity_class] / 4;
                            cb_prog_width <= prog_width[sensitivity_class] / 4;
                        end
                        cb_prog_polarity <= 1'b0; // Positive
                        state <= ST_PROG_PULSE;
                    end
                end
                
                ST_COMPENSATION: begin
                    // Apply compensation pulse for overshoot
                    cb_prog_en <= 1'b1;
                    if (timer >= cb_prog_width / CLK_PERIOD_NS) begin
                        cb_prog_en <= 1'b0;
                        state <= ST_PROG_VERIFY;
                        timer <= 32'd0;
                    end else begin
                        timer <= timer + 1;
                    end
                end
                
                ST_FINAL_VERIFY: begin
                    // Final array-level verification
                    cb_read_en <= 1'b1;
                    if (cb_read_valid) begin
                        cb_read_en <= 1'b0;
                        if ((cb_read_data > g_target ? (cb_read_data - g_target) : (g_target - cb_read_data)) 
                            < ((G_MAX_US - G_MIN_US) * 78 / 10000)) begin
                            status <= 2'b00; // Success
                        end else begin
                            status <= 2'b10; // Verify fail
                        end
                        state <= ST_DONE;
                    end
                end
                
                ST_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= ST_IDLE;
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule