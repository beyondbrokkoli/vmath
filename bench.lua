-- ========================================================================
-- MODULES/bench.lua
-- Pure FFI-backed benchmarking. Zero Lua table thrashing in the hot loop.
-- ========================================================================
local ffi = require("ffi")

ffi.cdef[[
    typedef struct {
        double total;
        double min;
        double max;
        int count;
        float start_time;
    } BenchStat;
]]

local MAX_BENCH_SLOTS = 32
local BenchData = ffi.new("BenchStat[?]", MAX_BENCH_SLOTS)

for i = 0, MAX_BENCH_SLOTS - 1 do
    BenchData[i].min = 9999999.0
    BenchData[i].max = 0.0
end

local registry_map = {}
local next_id = 0

BENCH = {}

function BENCH.Run(label, func)
    -- Map string label to an integer ID (only allocates/looks up once!)
    local id = registry_map[label]
    if not id then
        id = next_id
        registry_map[label] = id
        next_id = next_id + 1
    end

    local start = love.timer.getTime()
    func()
    local duration = love.timer.getTime() - start

    -- Pure FFI pointer arithmetic. The JIT compiler loves this.
    BenchData[id].count = BenchData[id].count + 1
    BenchData[id].total = BenchData[id].total + duration
    if duration < BenchData[id].min then BenchData[id].min = duration end
    if duration > BenchData[id].max then BenchData[id].max = duration end
end

-- We need to reset min/max periodically, otherwise the 'max' just
-- gets stuck permanently on the 1-second lag spike from when the app started.
function BENCH.ResetRollingStats()
    for i = 0, next_id - 1 do
        BenchData[i].count = 0
        BenchData[i].total = 0
        BenchData[i].min = 9999999.0
        BenchData[i].max = 0.0
    end
end
function BENCH.Begin(label)
    local id = registry_map[label]
    if not id then 
        id = next_id; registry_map[label] = id; next_id = next_id + 1 
    end
    -- Store start time in a temporary Lua table or directly in the FFI struct
    BenchData[id].start_time = love.timer.getTime()
end

function BENCH.End(label)
    local id = registry_map[label]
    local duration = love.timer.getTime() - BenchData[id].start_time
    
    BenchData[id].count = BenchData[id].count + 1
    BenchData[id].total = BenchData[id].total + duration
    if duration < BenchData[id].min then BenchData[id].min = duration end
    if duration > BenchData[id].max then BenchData[id].max = duration end
end
function BENCH.PrintAndReset(label)
    local id = registry_map[label]
    if not id or BenchData[id].count == 0 then return end
    
    local d = BenchData[id]
    local avg_ms = (d.total / d.count) * 1000
    local min_ms = d.min * 1000
    local max_ms = d.max * 1000
    
    print(string.format("[BENCH: %s] Avg: %5.2f ms | Min: %5.2f ms | Max: %5.2f ms", label, avg_ms, min_ms, max_ms))
    
    -- Reset for the next batch so we don't carry old lag spikes
    d.count = 0
    d.total = 0
    d.min = 9999999.0
    d.max = 0.0
end
