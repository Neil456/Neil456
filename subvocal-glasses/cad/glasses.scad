// Radar silent-speech glasses — parametric frame (v0.1)
//
// Coordinates: X = across the face (+X = wearer's LEFT), Y = toward the back
// of the head (front of frame at Y = 0), Z = up (lens centres at Z = 0).
//
// Render one printable part at a time:
//   openscad -D 'part="front"' -o front.stl glasses.scad
// Parts: assembly | front | temple_left | temple_right | lid_left | lid_right
//        | pod_cdm324 | pod_xm125      (pods: set pod_side = "left" / "right")

/* [Part] */
part = "assembly";
pod_side = "left";          // which side a printed pod is for (yaw is mirrored)

/* [Face fit — measure yourself and adjust] */
lens_w = 50;                // lens opening width
lens_h = 36;                // lens opening height
bridge = 18;                // gap between lens openings (nose bridge width)
rim = 4;                    // rim width around lenses
front_t = 5;                // front frame thickness (front-to-back)
top_bar_h = 7;              // thickened top rim that carries the wire groove
endpiece_w = 8;             // block at each outer corner that holds a temple
socket_depth = 8;           // how far a temple slides into its endpiece
temple_len = 105;           // hinge to ear bend
temple_drop_len = 35;       // part that curls down behind the ear
temple_drop_angle = 30;     // degrees
temple_w = 4;
temple_h = 7;

/* [Electronics bays — inner sizes incl. clearance; measure your parts] */
// [length along temple, height, thickness]
bay_left  = [26, 19.0, 5.5];  // Seeed XIAO nRF52840 Sense (21 x 17.8 mm) + wires
bay_right = [36, 21.5, 4.8];  // LiPo 402030 (~180 mAh) incl. protection PCB
bay_wall = 1.6;
bay_offset = socket_depth + 6; // bay starts this far behind the temple's front end
lid_recess = 1.0;           // recess around the bay opening for the lid
lid_t = 1.2;
groove_w = 2.2;             // open wire groove (30 AWG silicone wire)
groove_d = 2.0;

/* [Sensor pods — module sizes are GUESSES, measure before printing] */
cdm324 = [25.5, 25.5, 8.0]; // board w, board h, thickness incl. can
xm125  = [25.8, 25.8, 6.0]; // SparkFun-style XM125 breakout (check yours)
pod_wall = 1.6;
pod_lip = 1.4;              // lip that keeps the module from falling out
pod_arm_len = 30;           // pivot hole to cradle centre (downward)
pod_out = 4;                // extra outward offset of the cradle, keeps it off the cheek
pod_yaw = 25;               // degrees the sensor face is turned toward the face

/* [Hardware] */
m2_hole = 2.3;
m2_nut_flats = 4.3;
m2_nut_t = 1.8;
tol = 0.3;                  // fit clearance for slots and sockets

$fn = 40;
eps = 0.01;

// ---------- derived dimensions ----------
lens_cx = bridge / 2 + lens_w / 2;            // lens centre X
x_outer = bridge / 2 + lens_w + rim;          // outer edge of rims
end_x   = x_outer + endpiece_w;               // outer edge of endpiece
top_z   = lens_h / 2 + top_bar_h;             // top of frame
zt0     = top_z - temple_h;                   // temple bottom (temple top = top_z)
temple_cx = x_outer + endpiece_w / 2;         // temple centre line X
socket_y0 = front_t + 1;                      // socket front (temple front end)
block_len = front_t + 1 + socket_depth;       // endpiece block length in Y
screw_y = socket_y0 + socket_depth / 2;       // temple cross-screw position
tab_h = 13;
tab_hole_z = zt0 - 3 - tab_h / 2 - 1;         // pod pivot hole height
tab_y = socket_y0 + socket_depth / 2;

// ---------- helpers ----------
module rrect(w, h, r) { offset(r) square([w - 2 * r, h - 2 * r], center = true); }

// 2D shape in XZ, extruded from Y = 0 to Y = t
module xz_extrude(t) { translate([0, t, 0]) rotate([90, 0, 0]) linear_extrude(t) children(); }

module hex_nut_trap(flats, t) { cylinder(d = flats / cos(30) + tol, h = t, $fn = 6); }

// ---------- front frame ----------
module front_outline() {
    difference() {
        union() {
            for (s = [-1, 1]) translate([s * lens_cx, 0]) rrect(lens_w + 2 * rim, lens_h + 2 * rim, 9);
            // top bar across the whole frame
            translate([-end_x, lens_h / 2]) square([2 * end_x, top_bar_h]);
            // endpieces
            for (s = [-1, 1]) mirror([s < 0 ? 1 : 0, 0])
                translate([x_outer - 2, zt0 - 3]) square([endpiece_w + 2, top_z - zt0 + 3]);
        }
        for (s = [-1, 1]) translate([s * lens_cx, 0]) rrect(lens_w, lens_h, 7);
    }
}

module nose_pads() {
    for (s = [-1, 1]) hull() {
        translate([s * (bridge / 2 + 1.5), front_t - 1, -2]) cube([3, 1, 10], center = true);
        translate([s * (bridge / 2 + 3.5), front_t + 6, -8]) rotate([0, s * 25, 0])
            scale([0.45, 1, 1]) cylinder(d = 10, h = 2, center = true);
    }
}

module endpiece_block(s) {
    mirror([s < 0 ? 1 : 0, 0, 0]) difference() {
        union() {
            translate([x_outer, 0, zt0 - 3]) cube([endpiece_w, block_len, temple_h + 3]);
            // pod mounting tab, flush with the outer face, hanging below the block
            hull() {
                translate([end_x - 3, tab_y - 5, zt0 - 3 - eps]) cube([3, 10, eps]);
                translate([end_x - 3, tab_y, tab_hole_z]) rotate([0, 90, 0]) cylinder(d = 10, h = 3);
            }
        }
        // U-socket for the temple (open on top so the wire groove passes through)
        translate([temple_cx - (temple_w + tol) / 2, socket_y0, zt0 - tol / 2])
            cube([temple_w + tol, socket_depth + eps, temple_h + 5]);
        // cross screw through block + temple
        translate([x_outer - 1, screw_y, zt0 + temple_h / 2]) rotate([0, 90, 0]) cylinder(d = m2_hole, h = endpiece_w + 2);
        // pod pivot hole
        translate([end_x - 4, tab_y, tab_hole_z]) rotate([0, 90, 0]) cylinder(d = m2_hole, h = 6);
    }
}

module front() {
    difference() {
        union() {
            xz_extrude(front_t) front_outline();
            nose_pads();
            for (s = [-1, 1]) endpiece_block(s);
        }
        // wire groove on the rear face of the top bar...
        translate([-temple_cx, front_t - groove_d, top_z - temple_h / 2 - groove_w / 2])
            cube([2 * temple_cx, groove_d + eps, groove_w]);
        // ...turning back along the top of each endpiece into the socket
        for (s = [-1, 1])
            translate([s * temple_cx - groove_w / 2, front_t - groove_d, top_z - temple_h / 2 - groove_w / 2])
                cube([groove_w, socket_y0 - front_t + groove_d + eps, temple_h / 2 + groove_w / 2 + eps]);
    }
}

// ---------- temples (built as the LEFT temple: local +X = outward) ----------
function bay_dims(left) = left ? bay_left : bay_right;

module temple_body(left) {
    b = bay_dims(left);
    bay_x0 = -temple_w / 2;
    bay_t = bay_wall + b[2] + lid_recess + 0.2;
    bay_hgt = b[1] + 2 * bay_wall;
    union() {
        translate([-temple_w / 2, 0, 0]) cube([temple_w, temple_len, temple_h]);
        // ear drop, joined smoothly with a hull
        hull() {
            translate([-temple_w / 2, temple_len - 6, 0]) cube([temple_w, 6, temple_h]);
            translate([0, temple_len, temple_h]) rotate([-temple_drop_angle, 0, 0])
                translate([-temple_w / 2, 0, -temple_h]) cube([temple_w, 6, temple_h]);
        }
        translate([0, temple_len, temple_h]) rotate([-temple_drop_angle, 0, 0]) hull() {
            translate([-temple_w / 2, 0, -temple_h]) cube([temple_w, eps, temple_h]);
            translate([0, temple_drop_len, -temple_h / 2]) rotate([0, 90, 0]) cylinder(d = temple_h, h = temple_w, center = true);
        }
        // electronics bay: flat on the head side, bulging outward, top flush with temple
        hull() {
            translate([bay_x0, bay_offset, temple_h - bay_hgt]) cube([bay_t, b[0] + 2 * bay_wall, bay_hgt]);
            // taper back into the temple so it does not snag
            translate([bay_x0, bay_offset - 6, 0]) cube([temple_w, b[0] + 2 * bay_wall + 12, temple_h]);
        }
    }
}

module temple(left = true) {
    b = bay_dims(left);
    cav_x0 = -temple_w / 2 + bay_wall;
    cav_y0 = bay_offset + bay_wall;
    cav_z0 = temple_h - bay_wall - b[1];
    outer_x = cav_x0 + b[2] + lid_recess + 0.2;
    difference() {
        temple_body(left);
        // bay cavity, open on the outward face
        translate([cav_x0, cav_y0, cav_z0]) cube([b[2] + 10, b[0], b[1]]);
        // lid recess
        translate([outer_x - lid_t, cav_y0 - lid_recess, cav_z0 - lid_recess])
            cube([lid_t + 5, b[0] + 2 * lid_recess, b[1] + 2 * lid_recess]);
        // wire groove along the top, from the front end into the bay
        translate([-groove_w / 2, -eps, temple_h - groove_d]) cube([groove_w, cav_y0 + 3, groove_d + eps]);
        translate([0, cav_y0 + 1.5, cav_z0 + b[1] - 1]) cylinder(d = groove_w + 0.4, h = bay_wall + 3);
        // cross-screw hole that lines up with the endpiece
        translate([-temple_w, socket_depth / 2, temple_h / 2]) rotate([0, 90, 0]) cylinder(d = m2_hole, h = 2 * temple_w);
        if (left) {
            // USB-C opening in the rear wall (XIAO components face outward, USB to the rear)
            translate([cav_x0 + 2.6, cav_y0 + b[0] - eps, cav_z0 + b[1] / 2])
                rotate([-90, 0, 0]) hull() for (dz = [-2.9, 2.9]) translate([0, dz, 0]) cylinder(d = 4.2, h = bay_wall + 2);
            // sensor cable entry through the front wall, near the bottom
            translate([cav_x0 + b[2] / 2, bay_offset - 1, cav_z0 + 3]) rotate([-90, 0, 0]) cylinder(d = 3.6, h = bay_wall + 2);
        } else {
            // slot in the bay floor for a mini slide switch actuator (battery on/off)
            translate([cav_x0 + 1, cav_y0 + b[0] - 9, cav_z0 - bay_wall - 1]) cube([2.8, 5.5, bay_wall + 2]);
        }
    }
}

module lid(left = true) {
    b = bay_dims(left);
    union() {
        cube([b[0] + 2 * lid_recess - tol, b[1] + 2 * lid_recess - tol, lid_t]);
        // friction plug that presses into the cavity opening
        translate([lid_recess + tol / 2, lid_recess + tol / 2, lid_t - eps])
            difference() {
                cube([b[0] - 2 * tol, b[1] - 2 * tol, 1.2]);
                translate([1.2, 1.2, -eps]) cube([b[0] - 2 * tol - 2.4, b[1] - 2 * tol - 2.4, 2]);
            }
    }
}

// ---------- sensor pods (built for the LEFT side; pivot hole on the X axis) ----------
module pod_cradle(m) {
    ow = m[0] + 2 * pod_wall; oh = m[1] + 2 * pod_wall; ot = m[2] + 2 * pod_wall;
    // module plane = XY, antenna faces -Z, slides in from +Y
    difference() {
        translate([-ow / 2, -oh / 2, -ot / 2]) cube([ow, oh, ot]);
        // module pocket, open toward +Y
        translate([-(m[0] + tol) / 2, -(m[1] + tol) / 2, -(m[2] + tol) / 2]) cube([m[0] + tol, m[1] + 10, m[2] + tol]);
        // antenna window (bottom), leaving a lip
        translate([-(m[0] - 2 * pod_lip) / 2, -(m[1] - 2 * pod_lip) / 2, -ot]) cube([m[0] - 2 * pod_lip, m[1] + 10, ot]);
        // slot in the back for pins / cable
        translate([-(m[0] - 6) / 2, -(m[1]) / 2 + 2, 0]) cube([m[0] - 6, m[1] + 10, ot]);
    }
}

module pod(m) {
    plate = [3, 11, 11];
    cradle_z = -pod_arm_len;
    ot = m[2] + 2 * pod_wall;
    // cradle sits OUTSIDE the pivot plate so its inner edge clears the cheek
    cradle_x = (m[0] + 2 * pod_wall) / 2 * cos(pod_yaw) + pod_out;
    difference() {
        union() {
            // pivot plate, sits against the outer face of the frame tab
            hull() {
                rotate([0, 90, 0]) cylinder(d = plate[1], h = plate[0]);
                translate([0, -plate[1] / 2, -plate[2] / 2 - 4]) cube([plate[0], plate[1], eps]);
            }
            // arm down to the cradle
            hull() {
                translate([0, -plate[1] / 2, -plate[2] / 2 - 4]) cube([plate[0], plate[1], eps]);
                translate([cradle_x, 0, cradle_z]) rotate([0, pod_yaw, 0]) translate([0, 0, ot / 2 - 0.5])
                    cube([10, 10, 1], center = true);
            }
            translate([cradle_x, 0, cradle_z]) rotate([0, pod_yaw, 0]) pod_cradle(m);
        }
        // pivot screw + nut trap on the outer face
        rotate([0, 90, 0]) translate([0, 0, -1]) cylinder(d = m2_hole, h = plate[0] + 2);
        translate([plate[0] - m2_nut_t + eps, 0, 0]) rotate([0, 90, 0]) rotate([0, 0, 30]) hex_nut_trap(m2_nut_flats, m2_nut_t);
        // keep the module pocket clear of the arm
        translate([cradle_x, 0, cradle_z]) rotate([0, pod_yaw, 0])
            translate([-(m[0] + tol) / 2, -(m[1] + tol) / 2, -(m[2] + tol) / 2]) cube([m[0] + tol, m[1] + 10, m[2] + tol]);
    }
}

module pod_for_side(m, side) { if (side == "right") mirror([1, 0, 0]) pod(m); else pod(m); }

// ---------- assembly ----------
module placed_temple(s) {
    translate([s * temple_cx, socket_y0, zt0]) mirror([s < 0 ? 1 : 0, 0, 0]) temple(s > 0);
}

module assembly() {
    color("DimGray") front();
    color("SlateGray") for (s = [-1, 1]) placed_temple(s);
    // left pod: XM125 (wireless build); right pod: CDM324 (tethered experiments)
    color("Orange") translate([end_x, tab_y, tab_hole_z]) pod_for_side(xm125, "left");
    color("Gold") translate([-end_x, tab_y, tab_hole_z]) pod_for_side(cdm324, "right");
}

// ---------- part selector (printable parts are laid flat for printing) ----------
if (part == "assembly") assembly();
else if (part == "front") rotate([90, 0, 0]) front();                                        // front face down
else if (part == "temple_left") rotate([0, -90, 0]) translate([temple_w / 2, 0, 0]) temple(true);   // head side down
else if (part == "temple_right") mirror([1, 0, 0]) rotate([0, -90, 0]) translate([temple_w / 2, 0, 0]) temple(false);
else if (part == "lid_left") lid(true);
else if (part == "lid_right") lid(false);
// pods print standing on the closed (-Y) face of the cradle; enable supports under the pivot plate
else if (part == "pod_cdm324") translate([0, 0, cdm324[1] / 2 + pod_wall]) rotate([90, 0, 0]) pod_for_side(cdm324, pod_side);
else if (part == "pod_xm125") translate([0, 0, xm125[1] / 2 + pod_wall]) rotate([90, 0, 0]) pod_for_side(xm125, pod_side);
