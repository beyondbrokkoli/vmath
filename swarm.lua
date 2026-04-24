local bit = require("bit")
local Memory = require("memory")

return function()
    local Swarm = {}
    local PCOUNT = 10000
    local target_state = 0
    local gravity_blend = 1.0
    local space_pressed_last = false

    function Swarm.Init()
        local A = Memory.Arrays
        
        -- Setup Object ID 0 Base Data
        A.Obj_X[0], A.Obj_Y[0], A.Obj_Z[0] = 0, 0, 0
        A.Obj_Radius[0] = 999999
        A.Obj_FWX[0], A.Obj_FWY[0], A.Obj_FWZ[0] = 0, 0, 1
        A.Obj_RTX[0], A.Obj_RTY[0], A.Obj_RTZ[0] = 1, 0, 0
        A.Obj_UPX[0], A.Obj_UPY[0], A.Obj_UPZ[0] = 0, 1, 0
        
        A.Obj_VertStart[0] = 0
        A.Obj_VertCount[0] = PCOUNT * 4
        A.Obj_TriStart[0] = 0
        A.Obj_TriCount[0] = PCOUNT * 4

        -- Initialize Particles
        for i = 0, PCOUNT - 1 do
            A.Swarm_PX[i] = (math.random() - 0.5) * 20000
            A.Swarm_PY[i] = (math.random() - 0.5) * 10000 + 5000
            A.Swarm_PZ[i] = (math.random() - 0.5) * 20000
            A.Swarm_VX[i] = (math.random() - 0.5) * 5000
            A.Swarm_VY[i] = (math.random() - 0.5) * 5000
            A.Swarm_VZ[i] = (math.random() - 0.5) * 5000
            A.Swarm_Seed[i] = i / (PCOUNT - 1)
        end

        local tIdx = 0
        local col1 = bit.bor(0xFF000000, bit.lshift(255, 16), 0, 0)
        local col2 = bit.bor(0xFF000000, 0, bit.lshift(255, 8), 0)
        local col3 = bit.bor(0xFF000000, 0, 0, 255)
        local col4 = bit.bor(0xFF000000, 0, bit.lshift(255, 8), 255)

        for i = 0, PCOUNT - 1 do
            local base = i * 4
            A.Tri_V1[tIdx], A.Tri_V2[tIdx], A.Tri_V3[tIdx] = base+0, base+1, base+2; A.Tri_BakedColor[tIdx] = col1; tIdx = tIdx + 1
            A.Tri_V1[tIdx], A.Tri_V2[tIdx], A.Tri_V3[tIdx] = base+0, base+2, base+3; A.Tri_BakedColor[tIdx] = col2; tIdx = tIdx + 1
            A.Tri_V1[tIdx], A.Tri_V2[tIdx], A.Tri_V3[tIdx] = base+0, base+3, base+1; A.Tri_BakedColor[tIdx] = col3; tIdx = tIdx + 1
            A.Tri_V1[tIdx], A.Tri_V2[tIdx], A.Tri_V3[tIdx] = base+1, base+3, base+2; A.Tri_BakedColor[tIdx] = col4; tIdx = tIdx + 1
        end
    end

    function Swarm.Tick(dt)
        local space_down = love.keyboard.isDown("space")
        if space_down and not space_pressed_last then
            target_state = target_state + 1
            if target_state > 4 then target_state = 0 end -- Capped at 4 because 5 & 6 (Metal/Paradox) were removed from the testbed!
        end
        space_pressed_last = space_down

        if target_state == 0 then gravity_blend = math.min(1.0, gravity_blend + dt * 2.0)
        else gravity_blend = math.max(0.0, gravity_blend - dt * 2.0) end

        -- Pass state parameters directly into the C-Struct
        local mem = Memory.RenderStruct
        mem.Swarm_State = target_state
        mem.Swarm_GravityBlend = gravity_blend
        mem.Swarm_Explode1 = love.mouse.isDown(1)
        mem.Swarm_Explode2 = love.mouse.isDown(2)
    end

    return Swarm
end
