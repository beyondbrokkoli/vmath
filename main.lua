local ffi = require("ffi")
local bit = require("bit")
local VibeMath = require("load")
local Memory = require("memory")
local Sequence = require("sequence")

local CANVAS_W, CANVAS_H
local ScreenBuffer, ScreenImage, ScreenPtr
local ZBuffer

local global_time = 0
local CMD = { CLEAR = 1, SWARM_TICK = 2, SPHERE_TICK = 3, RENDER_CULL = 4, RENDER_TWOTONE = 5 }

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

    q[q_len] = CMD.CLEAR;       q_len = q_len + 1  
    q[q_len] = CMD.SWARM_TICK;  q_len = q_len + 1  
    q[q_len] = CMD.RENDER_CULL; q_len = q_len + 1  
    q[q_len] = 0;               q_len = q_len + 1 -- Render ID 0 (Swarm)

    VibeMath.vmath_execute_queue(
        q, q_len, MainCamera, Memory.RenderStruct, 
        ScreenPtr, ZBuffer, CANVAS_W, CANVAS_H, 
        global_time, love.timer.getDelta()
    )

    ScreenImage:replacePixels(ScreenBuffer)
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.setBlendMode("replace")
    love.graphics.draw(ScreenImage, 0, 0)
    love.graphics.setBlendMode("alpha")
    love.graphics.setColor(0, 1, 0, 1)
    love.graphics.print("FPS: " .. love.timer.getFPS() .. " | SWARM ISOLATION TEST", 10, 10)
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
