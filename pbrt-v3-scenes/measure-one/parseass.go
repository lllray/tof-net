/*
how are we going to deal with emission??
how do we find area light sources?
when multiple shaders bound to mesh, are we getting that right?
just do a full translator, including geometry
other shader types needed? ignore geom stuff, warn about unknown nodes...
support ramps
do we need a new pbrt material to properly support this?
  or, maybe an 'add' material that takes an array of named materials and adds them together?
*/
package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"strconv"
	"strings"
)

var lineno int
var wTextures, wMaterials, wGeometry, wLights, wMain *bufio.Writer

func fatal(s string) {
	fmt.Fprintf(os.Stderr, "%d: %s\n", lineno, s)
	panic(s)
	os.Exit(1)
}

func check(b bool, s string) {
	if !b {
		fatal(s)
	}
}

func parseBlock(r *bufio.Reader, ismesh bool) map[string]string {
	m := make(map[string]string)

	s, err := r.ReadString('\n')
	lineno += 1
	if err != nil {
		fatal(fmt.Sprintf("%v", err))
	}
	if strings.TrimSpace(s) != "{" {
		fatal("didn't find expected opening brace for block")
	}

	for {
		s, err := r.ReadString('\n')
		lineno += 1
		if err != nil {
			fatal(fmt.Sprintf("%v", err))
		}
		s = strings.TrimSpace(s)
		if s == "}" {
			return m
		}
		if s == "" {
			continue
		}

		nextra := 0
		if ismesh && len(s) > 4 &&
			(s[:5] == "vlist" || s[:5] == "nlist" || s[:6] == "uvlist" ||
				s[:5] == "vidxs" || s[:5] == "nidxs" || s[:6] == "uvidxs" ||
				s[:6] == "shidxs" ||
				s[:6] == "nsides") {
			nextra = 1
		}
		if ismesh && s == "matrix" {
			// TODO: are they always this way, or is this just how the c4d
			// exporter does them?  More generally, are newlines allowed
			// anywhere whitespace is?
			nextra = 4
		}
		for i := 0; i < nextra; i += 1 {
			next, err := r.ReadString('\n')
			lineno += 1
			if err != nil {
				fatal(fmt.Sprintf("%v", err))
			}
			s += " " + strings.TrimSpace(next)
		}

		if ismesh {
			log.Printf("line: %s\n", s)
		}

		nv := strings.SplitAfterN(s, " ", 2)
		m[strings.TrimSpace(nv[0])] = strings.TrimSpace(nv[1])
	}
}

func skipBlock(r *bufio.Reader) {
	// Hack: just search for closing brace
	lastnew := false
	for {
		c, err := r.ReadByte()
		if err != nil {
			fatal(fmt.Sprintf("skipBlock error %+v", err))
		}
		if c == '}' && lastnew {
			return
		}
		lastnew = c == '\n'
		if lastnew {
			lineno += 1
		}
	}
}

func getFileWriterOrExit(p string) *bufio.Writer {
	f, err := os.Create(p)
	if err != nil {
		fatal(fmt.Sprintf("%s: %+v", p, err))
	}
	return bufio.NewWriter(f)
}

func main() {
	wTextures = getFileWriterOrExit("textures.pbrt")
	wMaterials = getFileWriterOrExit("materials.pbrt")
	// wGeometry = getFileWriterOrExit("geometry.pbrt")
	// wLights = getFileWriterOrExit("lights.pbrt")
	//wMain = getFileWriterOrExit("main.pbrt")

	r := bufio.NewReader(os.Stdin)

	// Parse the file; build up maps of materials and textures
	var mtls []map[string]string
	var images []map[string]string
	ignoredNodes := make(map[string]bool)

	for {
		s, err := r.ReadString('\n')
		lineno += 1

		if err == io.EOF {
			break
		} else if err != nil {
			fatal(fmt.Sprintf("parseass: %v", err))
		}

		scan := bufio.NewScanner(strings.NewReader(s))
		for scan.Scan() {
			cmd := strings.TrimSpace(scan.Text())
			if cmd == "" {
				continue
			} else if cmd == "standard" {
				mtls = append(mtls, parseBlock(r, false))
			} else if cmd == "image" {
				images = append(images, parseBlock(r, false))
			} else if cmd == "polymesh" {
				//emitMesh(parseBlock(r, true))
				skipBlock(r)
			} else if cmd[0] == '#' {
				// comment
				//fmt.Printf("%s\n", cmd)
			} else {
				if _, ok := ignoredNodes[cmd]; !ok {
					log.Printf("ignoring node @ line %d \"%s\"\n", lineno, cmd)
					ignoredNodes[cmd] = true
				}
				skipBlock(r)
			}
		}
		if scan.Err() != nil {
			fatal(fmt.Sprintf("parseass: scanner error %v", scan.Err()))
		}
	}

	// First emit textures
	for _, img := range images {
		name := img["name"]
		fn := strings.Trim(img["filename"], "\"")

		// FIXME HACK
		fn = path.Join("textures", fn)

		if true /* allPNG */ {
			ext := path.Ext(fn)
			if ext == ".jpg" || ext == ".JPG" || ext == ".tif" || ext == ".tiff" ||
				ext == ".TIFF" || ext == ".TIF" {
				fn = fn[:len(fn)-len(ext)] + ".png"
			}
		}
		// Emit both float and rgb variants
		fmt.Fprintf(wTextures, "Texture \"%s-float\" \"float\" \"imagemap\"\n", name)
		fmt.Fprintf(wTextures, "  \"string filename\" \"%s\"\n", fn)
		fmt.Fprintf(wTextures, "Texture \"%s-color\" \"color\" \"imagemap\"\n", name)
		fmt.Fprintf(wTextures, "  \"string filename\" \"%s\"\n", fn)
	}

	// Emit pbrt materials
	// https://support.solidangle.com/display/NodeRef/standard
	for _, m := range mtls {
		names := strings.Split(m["name"], "|")
		name := strings.Replace(names[1], "_", " ", -1) // TODO: do this or not?

		fmt.Fprintf(wMaterials, "MakeNamedMaterial \"%s\"\n", name)
		fmt.Fprintf(wMaterials, "  \"string type\" \"substrate\"\n")

		//		if m["emission"] != "0" {
		//			fmt.Fprintf(wMaterials, "    \"rgb Kd\" [1000 1000 1000]\n")
		//			continue
		//		}

		emitColorMaterialParameter("Kd", m, name)
		// TODO: "diffuse_roughness" [0,1] -> OrenNayar. Need to add to uber mtl?
		emitColorMaterialParameter("Ks", m, name)
		emitFloatMaterialParameter("specular_roughness", "uroughness", m, name)
		emitFloatMaterialParameter("specular_roughness", "vroughness", m, name)
		// specular_anisotropy
		// emitColorMaterialParameter("Kr", m, name)
		//emitColorMaterialParameter("Kt", m, name)
		// refraction_roughness
		// transmittance
		// Ior
		// Fresnel stuff?
		// emisssion?
		// Ksss ?
		// bump mapping

		// Handle opacity specially...
		if _, err := getFloat3(m, "opacity", name); err != nil {
			// Ignore constant RGB opacity, which is always ~1 for this scene
			fmt.Fprintf(wMaterials, "  \"texture alpha\" \"%s-rgb\"\n", m["opacity"])
		}
	}

	if err := wTextures.Flush(); err != nil {
		fatal(fmt.Sprintf("textures.pbrt: %+v", err))
	}
	if err := wMaterials.Flush(); err != nil {
		fatal(fmt.Sprintf("materials.pbrt: %+v", err))
	}
	/*
		if err := wMain.Flush(); err != nil {
			fatal(fmt.Sprintf("main.pbrt: %+v", err))
		}
		if err := wLights.Flush(); err != nil {
			fatal(fmt.Sprintf("lights.pbrt: %+v", err))
		}
		if err := wGeometry.Flush(); err != nil {
			fatal(fmt.Sprintf("geometry.pbrt: %+v", err))
		}
	*/
}

func emitFloatMaterialParameter(param string, emitParam string, m map[string]string, blockname string) {
	val, err := getFloat(m, param, blockname)
	if err == nil {
		fmt.Fprintf(wMaterials, "  \"float %s\" %f\n", emitParam, val/10.)
	} else {
		fmt.Fprintf(wMaterials, "  \"texture %s\" \"%s-float\"\n", emitParam, m[param])
	}
}

func emitColorMaterialParameter(param string, m map[string]string, blockname string) {
	scale, errs := getFloat(m, param, blockname)
	f3, err3 := getFloat3(m, param+"_color", blockname)
	if errs == nil {
		// Scale is a constant
		if err3 == nil {
			// It's all constants. Multiply through and return a straight up RGB.
			fmt.Fprintf(wMaterials, "  \"rgb %s\" [ %f %f %f ]\n", param, f3[0]*scale, f3[1]*scale, f3[2]*scale)
		} else if scale == 1 {
			// Special case scale==1 just to keep things clean.
			fmt.Fprintf(wMaterials, "  \"texture %s\" \"%s-color\"\n", param, m[param+"_color"])
		}
	} else {
		// One or more textures.
		texname := fmt.Sprintf("%s-%s", blockname, param)
		fmt.Fprintf(wTextures, "Texture \"%s\" \"color\" \"mix\" \"rgb tex1\" [ 0 0 0 ]\n", texname)
		if errs == nil {
			fmt.Fprintf(wTextures, "  \"float amount\" %f\n", scale)
		} else {
			fmt.Fprintf(wTextures, "  \"texture amount\" \"%s-float\"\n", m[param])
		}
		if err3 == nil {
			fmt.Fprintf(wTextures, "  \"rgb tex2\" [ %f %f %f ]\n", f3[0], f3[1], f3[2])
		} else {
			fmt.Fprintf(wTextures, "  \"texture tex2\" \"%s-color\"\n", m[param+"_color"])
		}
		fmt.Fprintf(wMaterials, "  \"texture %s\" \"%s\"\n", param, texname)
	}
}

func getFloat(m map[string]string, name string, blockname string) (float64, error) {
	v, ok := m[name]
	if !ok {
		fatal(fmt.Sprintf("%s: didn't find a value for \"%s\"", blockname, name))
	}
	return strconv.ParseFloat(v, 32)
}

func getFloat3(m map[string]string, name string, blockname string) ([3]float64, error) {
	var f3 [3]float64
	v, ok := m[name]
	if !ok {
		return f3, fmt.Errorf("%s: didn't find a value for \"%s\"", blockname, name)
	}

	f := strings.Fields(v)
	if len(f) != 3 {
		return f3, fmt.Errorf("%s: didn't find three values after \"%s\"", blockname, name)
	}

	for i := 0; i < 3; i += 1 {
		val, err := strconv.ParseFloat(f[i], 32)
		if err != nil {
			return f3, fmt.Errorf("%s: float parse error for \"%s\": %v", blockname,
				f[i], err)
		}
		f3[i] = val
	}
	return f3, nil
}

func emitMesh(m map[string]string) {
	// wGeometry
}
