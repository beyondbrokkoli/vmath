local ffi = require("ffi")
local bit = require("bit")
local VibeMath = require("load")
local Memory = require("memory")
local Sequence = require("sequence") -- Bringing the DJ back!

local CANVAS_W, CANVAS_H
local ScreenBuffer, ScreenImage, ScreenPtr
local ZBuffer

local global_time = 0

-- OPCODES (Must match vibemath.c)
local CMD = {
    CLEAR = 1,
    SWARM_TICK = 2,
    SPHERE_TICK = 3,
    RENDER_CULL = 4,
}

function love.load()
    CANVAS_W, CANVAS_H = love.graphics.getPixelDimensions()
    MainCamera.fov = (CANVAS_W / 800) * 600

    ScreenBuffer = love.image.newImageData(CANVAS_W, CANVAS_H)
    ScreenImage = love.graphics.newImage(ScreenBuffer)
    ScreenPtr = ffi.cast("uint32_t*", ScreenBuffer:getPointer())
    ZBuffer = ffi.new("float[?]", CANVAS_W * CANVAS_H)

    -- 1. Load Modules (Sequence will automatically instantiate them)
    Sequence.LoadModule("camera", MainCamera)
    Sequence.LoadModule("swarm")

    -- 2. Run Module Initializers
    -- Swarm will claim Object ID 0 and Geometry block 0!
    Sequence.RunPhase("Init")

    -- 3. Setup the Testbed Sphere
    -- We dynamically claim the NEXT available object and geometry block!
    local id, _ = Memory.ClaimObjects(1) -- Will be ID 1
    local lats, longs = 100, 100
    local vStart, tStart = Memory.ClaimGeometry((lats + 1) * (longs + 1), lats * longs * 2)

    local A = Memory.Arrays
    A.Obj_X[id] = 0; A.Obj_Y[id] = 3000; A.Obj_Z[id] = 0
    A.Obj_Radius[id] = 7000
    A.Obj_FWX[id], A.Obj_FWY[id], A.Obj_FWZ[id] = 0, 0, 1
    A.Obj_RTX[id], A.Obj_RTY[id], A.Obj_RTZ[id] = 1, 0, 0
    A.Obj_UPX[id], A.Obj_UPY[id], A.Obj_UPZ[id] = 0, 1, 0
    
    A.Obj_VertStart[id] = vStart
    A.Obj_VertCount[id] = (lats + 1) * (longs + 1)
    A.Obj_TriStart[id] = tStart
    A.Obj_TriCount[id] = lats * longs * 2

    -- Wire the Triangle Indices for the Sphere
    local tIdx = tStart
    local col_white = bit.bor(0xFF000000, bit.lshift(200, 16), bit.lshift(200, 8), 200)
    for i = 0, lats - 1 do
        for j = 0, longs - 1 do
            -- Note: We add vStart to offset the indices behind the Swarm!
            local a = vStart + (i * (longs + 1)) + j
            local b = vStart + (i * (longs + 1)) + j + 1
            local c = vStart + ((i + 1) * (longs + 1)) + j + 1
            local d = vStart + ((i + 1) * (longs + 1)) + j

            A.Tri_V1[tIdx] = a; A.Tri_V2[tIdx] = d; A.Tri_V3[tIdx] = c; A.Tri_BakedColor[tIdx] = col_white; tIdx = tIdx + 1
            A.Tri_V1[tIdx] = a; A.Tri_V2[tIdx] = c; A.Tri_V3[tIdx] = b; A.Tri_BakedColor[tIdx] = col_white; tIdx = tIdx + 1
        end
    end
end

function love.update(dt)
    dt = math.min(dt, 0.033)
    global_time = global_time + dt
    
    -- Dispatch Camera and Swarm updates
    Sequence.RunPhase("Tick", dt)
    
    -- Animate the Sphere so we know it's alive
    local id = 1
    local A = Memory.Arrays
    A.Obj_FWX[id] = math.sin(global_time * 0.5)
    A.Obj_FWZ[id] = math.cos(global_time * 0.5)
    A.Obj_RTX[id] = math.cos(global_time * 0.5)
    A.Obj_RTZ[id] = -math.sin(global_time * 0.5)
end

function love.draw()
    local q = Memory.Arrays.CommandQueue
    local q_len = 0

    -- ========================================================
    -- BUILD THE COMMAND QUEUE
    -- ========================================================
    q[q_len] = CMD.CLEAR;       q_len = q_len + 1  

    -- Push Swarm commands
    q[q_len] = CMD.SWARM_TICK;  q_len = q_len + 1  
    q[q_len] = CMD.RENDER_CULL; q_len = q_len + 1  
    q[q_len] = 0;               q_len = q_len + 1 -- Argument: ID 0

    -- Push Sphere commands
    q[q_len] = CMD.SPHERE_TICK; q_len = q_len + 1  
    q[q_len] = CMD.RENDER_CULL; q_len = q_len + 1  
    q[q_len] = 1;               q_len = q_len + 1 -- Argument: ID 1

    -- ========================================================
    -- EXECUTE IN C (The only FFI crossing!)
    -- ========================================================
    VibeMath.vmath_execute_queue(
        q, q_len, 
        MainCamera, Memory.RenderStruct, 
        ScreenPtr, ZBuffer, CANVAS_W, CANVAS_H, 
        global_time, love.timer.getDelta()
    )

    ScreenImage:replacePixels(ScreenBuffer)
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.setBlendMode("replace")
    love.graphics.draw(ScreenImage, 0, 0)
    
    love.graphics.setBlendMode("alpha")
    love.graphics.setColor(0, 1, 0, 1)
    love.graphics.print("FPS: " .. love.timer.getFPS() .. " | SWARM + SPHERE CO-RENDER", 10, 10)
end

-- ========================================================
-- INPUT ROUTING
-- ========================================================
function love.keypressed(key)
    if key == "escape" then
        if love.mouse.getRelativeMode() then love.mouse.setRelativeMode(false) else love.event.quit() end
    end
    if key == "tab" then love.mouse.setRelativeMode(not love.mouse.getRelativeMode()) end
    Sequence.RunPhase("KeyPressed", key)
end

function love.mousemoved(x, y, dx, dy)
    Sequence.RunPhase("MouseMoved", x, y, dx, dy)
end

function love.mousepressed(x, y, button)
    if not love.mouse.getRelativeMode() then love.mouse.setRelativeMode(true) end
end
