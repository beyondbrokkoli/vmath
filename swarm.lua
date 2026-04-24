local ffi = require("ffi")
local bit = require("bit")
local VibeMath = require("load")
local RenderMeshFactory = require("render")

return function(Memory, MainCamera, Obj_X, Obj_Y, Obj_Z, Obj_Radius, Obj_FWX, Obj_FWY, Obj_FWZ, Obj_RTX, Obj_RTY, Obj_RTZ, Obj_UPX, Obj_UPY, Obj_UPZ, Obj_VertStart, Obj_VertCount, Obj_TriStart, Obj_TriCount, Vert_LX, Vert_LY, Vert_LZ, Vert_PX, Vert_PY, Vert_PZ, Vert_Valid, Tri_V1, Tri_V2, Tri_V3, Tri_BakedColor, Tri_Valid, Tri_ShadedColor)

    local Swarm = {}
    local swarm_obj_id
    local PCOUNT = 10000
    local VCOUNT = PCOUNT * 4
    local TCOUNT = PCOUNT * 4

    -- The Unified State Machine
    local target_state = 0 
    -- 0 = Floor Physics
    -- 1 = Bundle (Sphere)
    -- 2 = Galaxy
    -- 3 = Tornado
    -- 4 = Gyroscope
    -- 5 = Living Metal (Boiling Sphere)
    -- 6 = Smale's Paradox (Inside-Out Sphere)

    -- DOD Lerping Weights
    local gravity_blend = 1.0
    local metal_blend = 0.0
    local paradox_blend = 0.0
    -- scene_manager takeover
    -- local time_alive = 0.0
    local space_pressed_last = false

    -- Dedicated Physics Arrays
    local p_px = ffi.new("float[?]", PCOUNT)
    local p_py = ffi.new("float[?]", PCOUNT)
    local p_pz = ffi.new("float[?]", PCOUNT)
    local p_vx = ffi.new("float[?]", PCOUNT)
    local p_vy = ffi.new("float[?]", PCOUNT)
    local p_vz = ffi.new("float[?]", PCOUNT)
    local p_seed = ffi.new("float[?]", PCOUNT)

    local DrawMesh = RenderMeshFactory(
        Obj_X, Obj_Y, Obj_Z, Obj_Radius, Obj_FWX, Obj_FWY, Obj_FWZ, Obj_RTX, Obj_RTY, Obj_RTZ, Obj_UPX, Obj_UPY, Obj_UPZ,
        Obj_VertStart, Obj_VertCount, Obj_TriStart, Obj_TriCount,
        Vert_LX, Vert_LY, Vert_LZ, Vert_PX, Vert_PY, Vert_PZ, Vert_Valid,
        Tri_V1, Tri_V2, Tri_V3, Tri_BakedColor
    )

    function Swarm.Init()
        swarm_obj_id, _ = Memory.ClaimObjects(1)
        local id = swarm_obj_id
        local vStart, tStart = Memory.ClaimGeometry(VCOUNT, TCOUNT)

        Obj_X[id], Obj_Y[id], Obj_Z[id] = 0, 0, 0
        Obj_Radius[id] = 999999
        Obj_FWX[id], Obj_FWY[id], Obj_FWZ[id] = 0, 0, 1
        Obj_RTX[id], Obj_RTY[id], Obj_RTZ[id] = 1, 0, 0
        Obj_UPX[id], Obj_UPY[id], Obj_UPZ[id] = 0, 1, 0
        Obj_VertStart[id], Obj_VertCount[id] = vStart, VCOUNT
        Obj_TriStart[id], Obj_TriCount[id] = tStart, TCOUNT

        for i = 0, PCOUNT - 1 do
            p_px[i] = (math.random() - 0.5) * 20000
            p_py[i] = (math.random() - 0.5) * 10000 + 5000
            p_pz[i] = (math.random() - 0.5) * 20000
            p_vx[i] = (math.random() - 0.5) * 5000
            p_vy[i] = (math.random() - 0.5) * 5000
            p_vz[i] = (math.random() - 0.5) * 5000
            p_seed[i] = i / (PCOUNT - 1)
        end

        local tIdx = tStart
        local col1 = bit.bor(0xFF000000, bit.lshift(255, 16), 0, 0)
        local col2 = bit.bor(0xFF000000, 0, bit.lshift(255, 8), 0)
        local col3 = bit.bor(0xFF000000, 0, 0, 255)
        local col4 = bit.bor(0xFF000000, 0, bit.lshift(255, 8), 255)

        for i = 0, PCOUNT - 1 do
            local base = vStart + (i * 4)
            Tri_V1[tIdx], Tri_V2[tIdx], Tri_V3[tIdx] = base+0, base+1, base+2
            Tri_BakedColor[tIdx] = col1; tIdx = tIdx + 1
            Tri_V1[tIdx], Tri_V2[tIdx], Tri_V3[tIdx] = base+0, base+2, base+3
            Tri_BakedColor[tIdx] = col2; tIdx = tIdx + 1
            Tri_V1[tIdx], Tri_V2[tIdx], Tri_V3[tIdx] = base+0, base+3, base+1
            Tri_BakedColor[tIdx] = col3; tIdx = tIdx + 1
            Tri_V1[tIdx], Tri_V2[tIdx], Tri_V3[tIdx] = base+1, base+3, base+2
            Tri_BakedColor[tIdx] = col4; tIdx = tIdx + 1
        end
    end

    function Swarm.Tick(dt, current_time)
        -- scene_manager takeover
        --time_alive = time_alive + dt

        -- 1. Spacebar State Cycler (0 through 6)
        local space_down = love.keyboard.isDown("space")
        if space_down and not space_pressed_last then
            target_state = target_state + 1
            if target_state > 6 then target_state = 0 end
        end
        space_pressed_last = space_down

        -- 2. Smooth DOD Blending
        -- Gravity only rules in State 0
        if target_state == 0 then gravity_blend = math.min(1.0, gravity_blend + dt * 2.0)
        else gravity_blend = math.max(0.0, gravity_blend - dt * 2.0) end

        -- Metal only boils in State 5
        if target_state == 5 then metal_blend = math.min(1.0, metal_blend + dt * 0.5)
        else metal_blend = math.max(0.0, metal_blend - dt * 2.0) end

        -- Paradox only everts in State 6
        if target_state == 6 then paradox_blend = math.min(1.0, paradox_blend + dt * 0.5)
        else paradox_blend = math.max(0.0, paradox_blend - dt * 2.0) end

        -- 3. Explosions (Work in all states!)
        if love.mouse.isDown(1) then
            VibeMath.simd_apply_explosion(PCOUNT, p_px, p_py, p_pz, p_vx, p_vy, p_vz, 0, 5000, 0, 5000000.0 * dt, 15000.0)
        end
        if love.mouse.isDown(2) then
            VibeMath.simd_apply_explosion(PCOUNT, p_px, p_py, p_pz, p_vx, p_vy, p_vz, 0, 5000, 0, -4000000.0 * dt, 20000.0)
        end

        -- 4. KERNEL DISPATCH
        if gravity_blend > 0.0 then
            local cage = UniverseCage
            VibeMath.simd_update_physics_swarm(
                PCOUNT, p_px, p_py, p_pz, p_vx, p_vy, p_vz,
                cage.minX, cage.maxX, cage.minY, cage.maxY, cage.minZ, cage.maxZ,
                dt, -8000.0 * gravity_blend
            )
        end

        if gravity_blend < 1.0 then
            -- Determine which shape kernel to map target coordinates
            if target_state >= 1 and target_state <= 4 then
                VibeMath.simd_update_swarm_attractors(
                    PCOUNT, p_px, p_py, p_pz, p_vx, p_vy, p_vz, p_seed,
                    0, 5000, 0, current_time, dt, target_state
                )
            elseif target_state == 5 then
                VibeMath.simd_update_swarm_living_metal(
                    PCOUNT, p_px, p_py, p_pz, p_vx, p_vy, p_vz, p_seed,
                    0, 5000, 0, current_time, dt, metal_blend
                )
            elseif target_state == 6 then
                VibeMath.simd_update_swarm_paradox(
                    PCOUNT, p_px, p_py, p_pz, p_vx, p_vy, p_vz, p_seed,
                    0, 5000, 0, current_time, dt, paradox_blend
                )
            end
        end

        -- 5. Output Geometry
        local vStart = Obj_VertStart[swarm_obj_id]
        VibeMath.generate_swarm_geometry(
            PCOUNT, p_px, p_py, p_pz,
            Vert_LX + vStart, Vert_LY + vStart, Vert_LZ + vStart,
            120.0
        )
    end

    function Swarm.Raster(CANVAS_W, CANVAS_H, ScreenPtr, ZBuffer)
        DrawMesh(swarm_obj_id, swarm_obj_id, MainCamera, CANVAS_W, CANVAS_H, ScreenPtr, ZBuffer)
    end

    return Swarm
end
