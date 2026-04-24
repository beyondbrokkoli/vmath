local ffi = require("ffi")
local bit = require("bit")
local VibeMath = require("load")
local Memory = require("memory")
local Sequence = require("sequence")

local CANVAS_W, CANVAS_H
local ScreenBuffer, ScreenImage, ScreenPtr
local ZBuffer

local global_time = 0
local CMD = {
    CLEAR = 1,
    SWARM_APPLY_BASE_PHYSICS = 2,
    SWARM_APPLY_ATTRACTORS = 3,
    SWARM_APPLY_METAL = 4,
    SWARM_APPLY_PARADOX = 5,
    SWARM_GEN_QUADS = 6,
    SPHERE_TICK = 7,
    RENDER_CULL = 8,
    SWARM_EXPLOSION_PUSH = 9,   -- NEW
    SWARM_EXPLOSION_PULL = 10   -- NEW
}

function love.load()
    CANVAS_W, CANVAS_H = love.graphics.getPixelDimensions()
    MainCamera.fov = (CANVAS_W / 800) * 600

    ScreenBuffer = love.image.newImageData(CANVAS_W, CANVAS_H)
    ScreenImage = love.graphics.newImage(ScreenBuffer)
    ScreenPtr = ffi.cast("uint32_t*", ScreenBuffer:getPointer())
    ZBuffer = ffi.new("float[?]", CANVAS_W * CANVAS_H)

    Sequence.LoadModule("camera", MainCamera)
    Sequence.LoadModule("swarm")
    Sequence.RunPhase("Init")
    
    -- SPHERE TEMPORARILY DISABLED FOR TESTING
    -- local id, _ = Memory.ClaimObjects(1) ...
end

function love.update(dt)
    dt = math.min(dt, 0.033)
    global_time = global_time + dt
    Sequence.RunPhase("Tick", dt)
end

function love.draw()
    local q = Memory.Arrays.CommandQueue
    local q_len = 0
    local mem = Memory.RenderStruct

    -- 1. CLEAR BUFFERS
    q[q_len] = CMD.CLEAR; q_len = q_len + 1  

    -- 2. CONDITIONAL EXPLOSIONS (The Logic lives in Lua!)
    if love.mouse.isDown(1) then
        q[q_len] = CMD.SWARM_EXPLOSION_PUSH; q_len = q_len + 1
    end
    if love.mouse.isDown(2) then
        q[q_len] = CMD.SWARM_EXPLOSION_PULL; q_len = q_len + 1
    end

    -- 3. BASE PHYSICS (Always runs)
    q[q_len] = CMD.SWARM_APPLY_BASE_PHYSICS; q_len = q_len + 1

    -- 4. TARGET SHAPE KERNEL (Only queue the one we need)
    local state = mem.Swarm_State
    if state >= 1 and state <= 4 then
        q[q_len] = CMD.SWARM_APPLY_ATTRACTORS; q_len = q_len + 1
    elseif state == 5 then
        q[q_len] = CMD.SWARM_APPLY_METAL; q_len = q_len + 1
    elseif state == 6 then
        q[q_len] = CMD.SWARM_APPLY_PARADOX; q_len = q_len + 1
    end

    -- 5. GENERATE GEOMETRY
    q[q_len] = CMD.SWARM_GEN_QUADS; q_len = q_len + 1

    -- 6. RENDER THE SWARM
    q[q_len] = CMD.RENDER_CULL; q_len = q_len + 1  
    q[q_len] = 0;               q_len = q_len + 1 -- Pass ID 0 as argument

    -- ========================================================
    -- EXECUTE IN C (Zero logic, 100% throughput)
    -- ========================================================
    BENCH.Begin("CommandQueue_Execute")
    VibeMath.vmath_execute_queue(
        q, q_len, 
        MainCamera, mem, 
        ScreenPtr, ZBuffer, CANVAS_W, CANVAS_H, 
        global_time, love.timer.getDelta()
    )
    BENCH.End("CommandQueue_Execute")

    ScreenImage:replacePixels(ScreenBuffer)
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.setBlendMode("replace")
    love.graphics.draw(ScreenImage, 0, 0)
    
    love.graphics.setBlendMode("alpha")
    love.graphics.setColor(0, 1, 0, 1)
    love.graphics.print("FPS: " .. love.timer.getFPS() .. " | PURE COMMAND QUEUE", 10, 10)
end

function love.keypressed(key)
    if key == "escape" then
        if love.mouse.getRelativeMode() then love.mouse.setRelativeMode(false) else love.event.quit() end
    end
    if key == "tab" then love.mouse.setRelativeMode(not love.mouse.getRelativeMode()) end
    Sequence.RunPhase("KeyPressed", key)
end
function love.mousemoved(x, y, dx, dy) Sequence.RunPhase("MouseMoved", x, y, dx, dy) end
function love.mousepressed(x, y, button) if not love.mouse.getRelativeMode() then love.mouse.setRelativeMode(true) end end
